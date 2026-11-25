#!/usr/bin/env python3
"""Generate ShapeBlend images from precomputed PNG contour maps."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def blend_dataset(input_dir, contours_dir, output_dir, alpha, negative=False):
    input_dir = Path(input_dir).resolve()
    contours_dir = Path(contours_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if not 0.0 <= alpha <= 1.0:
        raise ValueError('Alpha must be between 0 and 1. Use --negative for inverted contours.')
    for directory in (input_dir, contours_dir):
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        if output_dir == directory or directory in output_dir.parents or output_dir in directory.parents:
            raise ValueError('Use a separate output directory outside the input and contour directories.')

    paths = sorted(p for p in input_dir.rglob('*')
                   if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError(f'No images found in {input_dir}')

    for index, path in enumerate(paths, 1):
        relative = path.relative_to(input_dir)
        contour_path = contours_dir / relative.with_suffix('.png')
        target = output_dir / relative

        with Image.open(path) as original, Image.open(contour_path) as contour:
            original = original.convert('L' if original.mode == 'L' else 'RGB')
            contour = contour.convert('L')
            if original.size != contour.size:
                raise ValueError(f'Image and contour sizes differ: {relative}')
            if negative:
                contour = ImageOps.invert(contour)
            contour = contour.convert(original.mode)
            blended = cv2.addWeighted(np.asarray(original), alpha,
                                      np.asarray(contour), 1.0 - alpha, 0)

        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(blended).save(target)

        if index % 1000 == 0 or index == len(paths):
            print(f'[{index}/{len(paths)}] Images processed', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Generate ShapeBlend images.')
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--contours-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--alpha', required=True, type=float,
                        help='Weight of the original image, between 0 and 1.')
    parser.add_argument('--negative', action='store_true',
                        help='Blend with the inverted contour map.')
    args = parser.parse_args()
    blend_dataset(args.input_dir, args.contours_dir, args.output_dir,
                  args.alpha, args.negative)


if __name__ == '__main__':
    main()
