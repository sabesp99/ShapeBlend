#!/usr/bin/env python3
"""Shape-Enhanced Voting (SEV) evaluation using precomputed ShapeBlend images."""
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.datasets.folder import default_loader

from eval import (make_parser, load_model, val_transform, dataset_path,
                  check_classes, make_loader, evaluate_benchmarks)

SEV_LEVELS = {'small': [1, 4, 7], 'medium': [1, 3, 5, 7, 9], 'large': list(range(1, 10))}


def get_levels(sev):
    values = SEV_LEVELS[sev]
    return ['clean'] + [f'blend{i}' for i in values] + [f'negative{i}' for i in values]


class OnlineDataset(torch.utils.data.Dataset):
    """Load aligned original and precomputed ShapeBlend views of each image."""
    def __init__(self, folders, transform=None):
        self.folders = [Path(p) for p in folders]
        self.transform = transform
        # The clean dataset defines the sample list and the class order.
        self.dataset = datasets.ImageFolder(self.folders[0])
        self.classes = self.dataset.classes
        self.root = self.dataset.root
        for folder in self.folders:
            if not folder.is_dir():
                raise FileNotFoundError(folder)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]
        relative = Path(path).relative_to(self.folders[0])
        imgs = []
        for folder in self.folders:
            image = default_loader(str(folder / relative))
            if self.transform:
                image = self.transform(image)
            imgs.append(image)
        return imgs, label


@torch.no_grad()
def test_loop(dataloader, model, device, levels, wc, mask=None):
    model.eval()
    correct = 0
    total = 0
    for imgs, labels in dataloader:
        labels = labels.to(device)
        aggregated_scores = None
        total_weight = 0
        for level, X in zip(levels, imgs):
            outputs = model(X.to(device))
            # Preserve the original A/R protocol: mask BEFORE softmax.
            if mask is not None:
                outputs = outputs[:, mask]
            scores = F.softmax(outputs, dim=1)
            weight = wc if level == 'clean' else 1
            if aggregated_scores is None:
                aggregated_scores = torch.zeros_like(scores)
            aggregated_scores += scores * weight
            total_weight += weight
        aggregated_scores /= total_weight
        correct += (aggregated_scores.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return 100 * correct / total


def main():
    parser = make_parser(__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--sev', choices=list(SEV_LEVELS), default='small')
    selection.add_argument('--lista', nargs='+', help='Custom list of levels, including clean')
    parser.add_argument('--wc', type=int, help='Clean image weight (default: number of blended views)')
    args = parser.parse_args()
    levels = args.lista if args.lista is not None else get_levels(args.sev)
    allowed = {'clean'} | {f'{prefix}{i}' for prefix in ('blend', 'negative') for i in range(1, 10)}
    if (len(levels) < 2 or 'clean' not in levels or len(set(levels)) != len(levels)
            or not set(levels) <= allowed):
        parser.error('--lista must include clean and distinct blend1..9 / negative1..9 levels')
    # Keep the original weighted accumulation order, while clean defines labels.
    if levels[0] != 'clean':
        parser.error('Specify clean as the first level in --lista')
    wc = args.wc if args.wc is not None else len(levels) - 1
    if wc < 1:
        parser.error('--wc must be positive')

    model, device, checkpoint = load_model(args)
    tag = args.sev if args.lista is None else '_'.join(levels[1:])
    output = (Path(__file__).resolve().parent / 'eval1000' /
              f'{args.exp}_{args.num}_sev_{tag}_wc{wc}.txt')
    output.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(benchmark, corruption=None, severity=None):
        folders = [dataset_path(args.data_dir, level, benchmark, corruption, severity)
                   for level in levels]
        dataset = OnlineDataset(folders, transform=val_transform(benchmark))
        mask = check_classes(dataset, benchmark)
        return test_loop(make_loader(dataset, args, device), model, device, levels, wc, mask)

    with output.open('w', encoding='utf-8') as file:
        def report(line):
            print(line, flush=True)
            print(line, file=file, flush=True)

        report(f'SEV evaluation | {args.exp} | Run {args.num} | {args.arch}')
        report(f'Checkpoint: {checkpoint}\nLevels: {", ".join(levels)}\nClean weight: {wc}')
        evaluate_benchmarks(args.datasets, evaluate, report)
        report('\nEvaluation complete.')
    print(f'Results saved to {output}')


if __name__ == '__main__':
    main()
