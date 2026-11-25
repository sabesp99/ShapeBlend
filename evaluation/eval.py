#!/usr/bin/env python3
"""Standard evaluation on ImageNet and its OOD benchmarks."""
import argparse
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import resnet50, convnext_tiny
from torchvision.transforms import InterpolationMode

from utils import all_wnids, imagenet_a_mask, imagenet_r_mask

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = {
    'val': 'ImageNet',
    'valC': 'ImageNet-C',
    'valB': 'ImageNet-C-bar',
    'valA': 'ImageNet-A',
    'valR': 'ImageNet-R',
}
CORRUPTIONS_C = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]
CORRUPTIONS_B = [
    'blue_noise_sample', 'brownish_noise', 'caustic_refraction',
    'checkerboard_cutout', 'cocentric_sine_waves', 'inverse_sparkles',
    'perlin_noise', 'plasma_noise', 'single_frequency_greyscale', 'sparkles',
]
SEVERITIES = range(1, 6)
MASKS = {'valA': imagenet_a_mask, 'valR': imagenet_r_mask}
# AlexNet mean error over the five severities, in CORRUPTIONS_C order.
# https://zenodo.org/records/2235448 (ImageNet-C benchmark)
# Full precision: https://openreview.net/pdf/547d35338f4b2e8f3f2cf42ad669c78f0e7bf1a0.pdf, Table 17.
ALEXNET_ERRORS = dict(zip(CORRUPTIONS_C, [
    0.886428, 0.894468, 0.922640, 0.819880, 0.826268,
    0.785948, 0.798360, 0.866816, 0.826572, 0.819324,
    0.564592, 0.853204, 0.646056, 0.717840, 0.606500,
]))


def make_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--exp', required=True, help='Training experiment identifier')
    parser.add_argument('--num', required=True, help='Training run identifier')
    parser.add_argument('--arch', choices=['resnet50', 'convnext_tiny'], default='convnext_tiny')
    parser.add_argument('--nclass', type=int, choices=[1000], default=1000)
    parser.add_argument('--cpus', type=int, default=32, help='DataLoader workers')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'imagenet1000')
    parser.add_argument('--checkpoint', type=Path, help='Override the default last checkpoint')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--datasets', nargs='+', choices=list(BENCHMARKS),
                        default=list(BENCHMARKS), help='Benchmarks to evaluate (default: all five)')
    return parser


def load_model(args):
    device = torch.device(args.device)
    model = resnet50(weights=None) if args.arch == 'resnet50' else convnext_tiny(weights=None)
    checkpoint = args.checkpoint or ROOT / 'train' / 'weights1000' / f'last{args.num}_{args.exp}.pth'
    state_dict = torch.load(checkpoint, map_location='cpu')
    # Support last checkpoints as well as raw best-model state dictionaries.
    state_dict = state_dict.get('model_state_dict', state_dict.get('state_dict', state_dict))
    state_dict = OrderedDict((k[7:] if k.startswith('module.') else k, v)
                             for k, v in state_dict.items())
    model.load_state_dict(state_dict)
    return model.to(device).eval(), device, checkpoint


def val_transform(benchmark):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    ops = []
    # The original corruption scripts use the images at their stored size.
    if benchmark not in ('valC', 'valB'):
        ops.extend([transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
                    transforms.CenterCrop(224)])
    ops.extend([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean, std)])
    return transforms.Compose(ops)


def dataset_path(root, level, benchmark, corruption=None, severity=None):
    folder = root / level / benchmark
    if corruption is None:
        return folder
    # Original flattened layout (e.g. gaussian_noise1), or nested layout.
    flat = folder / f'{corruption}{severity}'
    nested = folder / corruption / str(severity)
    return flat if flat.is_dir() else nested if nested.is_dir() else flat


def check_classes(dataset, benchmark):
    mask = MASKS.get(benchmark)
    expected = all_wnids if mask is None else [w for w, keep in zip(all_wnids, mask) if keep]
    if dataset.classes != expected:
        raise ValueError(f'Unexpected class folders or class order in {dataset.root}. '
                         f'Expected {len(expected)} ImageNet classes; found {len(dataset.classes)}.')
    return mask


def make_loader(dataset, args, device):
    return DataLoader(dataset, batch_size=args.batch, shuffle=False,
                      num_workers=args.cpus, pin_memory=device.type == 'cuda')


@torch.no_grad()
def test_loop(dataloader, model, device, mask=None):
    model.eval()
    correct = 0
    total = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        if mask is not None:
            pred = pred[:, mask]
        correct += (pred.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100 * correct / total


def evaluate_benchmarks(benchmarks, evaluate, report):
    """Shared reporting for standard evaluation and SEV."""
    summary = {}
    for benchmark in dict.fromkeys(benchmarks):
        name = BENCHMARKS[benchmark]
        report(f'\n--- {name} ---')
        if benchmark not in ('valC', 'valB'):
            acc = evaluate(benchmark)
            report(f'{name} Accuracy: {acc:.4f}%')
            summary[f'{name} Accuracy'] = acc
            continue

        corruptions = CORRUPTIONS_C if benchmark == 'valC' else CORRUPTIONS_B
        means, errors = [], []
        for c in corruptions:
            accuracies = [evaluate(benchmark, c, s) for s in SEVERITIES]
            ca = sum(accuracies) / len(accuracies)
            means.append(ca)
            # One line per corruption, with all severity accuracies and their mean.
            values = ' '.join(f'{acc:.4f}' for acc in accuracies)
            line = f'{c}: [{values}] | CA: {ca:.4f}%'
            if benchmark == 'valC':
                ce = (100 - ca) / ALEXNET_ERRORS[c]
                errors.append(ce)
                line += f' | CE: {ce:.4f}%'
            report(line)
        mca = sum(means) / len(means)
        report(f'{name} mCA: {mca:.4f}%')
        summary[f'{name} mCA'] = mca
        if errors:
            mce = sum(errors) / len(errors)
            report(f'{name} mCE: {mce:.4f}%')
            summary[f'{name} mCE'] = mce

    report('\n--- SUMMARY ---')
    for metric, value in summary.items():
        report(f'{metric}: {value:.4f}%')
    return summary


def main():
    parser = make_parser(__doc__)
    parser.add_argument('--subexp', default='clean', help='Dataset level to evaluate')
    args = parser.parse_args()
    model, device, checkpoint = load_model(args)
    output = Path(__file__).resolve().parent / 'eval1000' / f'{args.exp}_{args.num}_{args.subexp}_standard.txt'
    output.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(benchmark, corruption=None, severity=None):
        path = dataset_path(args.data_dir, args.subexp, benchmark, corruption, severity)
        dataset = datasets.ImageFolder(path, transform=val_transform(benchmark))
        mask = check_classes(dataset, benchmark)
        return test_loop(make_loader(dataset, args, device), model, device, mask)

    with output.open('w', encoding='utf-8') as file:
        def report(line):
            print(line, flush=True)
            print(line, file=file, flush=True)

        report(f'Standard evaluation | {args.exp} | Run {args.num} | {args.arch}')
        report(f'Checkpoint: {checkpoint}\nDataset level: {args.subexp}')
        evaluate_benchmarks(args.datasets, evaluate, report)
        report('\nEvaluation complete.')
    print(f'Results saved to {output}')


if __name__ == '__main__':
    main()
