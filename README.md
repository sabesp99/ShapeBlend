# ShapeBlend: Boosting Out-of-Distribution Robustness in Image Classification via Shape-based Blending Augmentation

Official repository for the paper:

**ShapeBlend: Boosting Out-of-Distribution Robustness in Image Classification via Shape-based Blending Augmentation**  
George Azzopardi, Sabatino Esposito, Antonio Greco, Mario Vento  
*Computer Vision and Image Understanding, 2026*  
DOI: [10.1016/j.cviu.2026.104768](https://doi.org/10.1016/j.cviu.2026.104768)

---

## Overview

ShapeBlend is a shape-based data augmentation strategy designed to improve out-of-distribution robustness in image classification.  
It enhances contour information by blending contour maps with the original image, encouraging more shape-aware representations.  
The method can also be used at inference time through **Shape-Enhanced Voting (SEV)**.

![ShapeBlend pipeline](Figures/pipeline.png)

---

## Summary of Results

Below is a concise summary of the main results reported in the paper.

| Method | Evaluation | IN-C mCA ↑ | IN-C̄ mCA ↑ | IN-R Acc. ↑ | IN-A Acc. ↑ |
|---|---|---:|---:|---:|---:|
| Baseline | Standard | 39.60 | 40.63 | 36.57 | 0.78 |
| SB | Standard | 43.61 | 42.30 | 38.90 | 1.96 |
| SB | SEV_S | 46.11 | 42.89 | 40.70 | 1.74 |
| AM | Standard | 46.60 | 44.52 | 40.15 | 1.14 |
| SB + AM | Standard | 49.76 | 45.33 | 44.12 | 2.48 |
| SB + AM | SEV_S | 51.35 | 45.29 | 45.56 | 2.39 |
| PR | Standard | 52.73 | 47.43 | 40.87 | 1.47 |
| SB + PR | Standard | 54.02 | 48.29 | 44.39 | 2.34 |
| SB + PR | SEV_S | 55.14 | 48.54 | 46.21 | 2.36 |
| DA | Standard | 52.96 | 48.83 | 42.78 | 3.34 |
| SB + DA | Standard | 55.08 | 49.43 | 46.27 | 5.53 |
| SB + DA | SEV_S | 56.51 | 49.40 | 47.62 | 5.12 |
| DA + AM | Standard | 57.98 | 51.09 | 45.67 | 3.85 |
| SB + DA + AM | Standard | 59.42 | 50.85 | 49.89 | 5.49 |
| SB + DA + AM | SEV_S | 59.77 | 50.56 | 50.97 | 5.27 |
| DA + PR | Standard | 59.59 | 52.12 | 45.02 | 4.26 |
| SB + DA + PR | Standard | 60.75 | 52.76 | 49.17 | **5.65** |
| SB + DA + PR | SEV_S | **61.11** | **52.76** | **50.60** | 5.46 |

SB: ShapeBlend; AM: AugMix; DA: DeepAugment; PR: PRIME; SEV_S: Shape-Enhanced Voting (Small configuration). All results are reported using ResNet-50 and averaged over three runs. Higher values indicate better performance.
For full experimental details, please refer to the paper.

---

## Pretrained Weights

The pretrained weights used in our experiments are
publicly available.

📦 [Download pretrained weights (Google Drive)](https://drive.google.com/drive/folders/11Ch3C26e-2PlgZ9Vvwq9inMWZwrkZpNA?usp=sharing)

## Training

We provide two training scripts covering the experimental
configurations presented in the paper:

- `train/train.py`: experiments without PRIME.
- `train/train_with_prime.py`: experiments involving PRIME.

### Dataset Preparation

Download ImageNet and organize the dataset as follows:

```text
imagenet1000/
└── clean/
    ├── train/
    │   ├── n01440764/
    │   └── ...
    └── val/
        ├── n01440764/
        └── ...
```

Place the `imagenet1000` directory in the repository root,
alongside the `train` directory.

### ShapeBlend

To reduce computational overhead across the numerous
experimental configurations, we precomputed the contour
maps and ShapeBlend variants offline.

During training, the blending factor is randomly sampled
online from the 18 available variants, while the original
images are retained in the training dataset.

#### Contour Extraction

We provide a GPU implementation of the Push-Pull CORF operator based on PyTorch.

First, navigate to the preprocessing directory:

```bash
cd preprocessing
```

Export the CORF receptive field geometry using MATLAB:

```matlab
export_corf_gpu_geometry('corf_geometry.mat', 2.2, 4);
```

This step only needs to be performed once and requires MATLAB with the Image Processing Toolbox.

Extract the contour maps using:

```bash
python corf_gpu.py \
    --geometry corf_geometry.mat \
    --input-dir ../imagenet1000/clean/train \
    --output-dir ../imagenet1000/contours/train \
    --inhibition-factor 1.8
```

Contour maps are saved as grayscale PNG images, preserving the original directory structure.

#### ShapeBlend Generation

Use `blend.py` to generate ShapeBlend images from the original images and their corresponding contour maps.

The `--alpha` argument specifies the absolute blending factor, while `--negative` enables blending with the complement of the contour map.

To generate all 18 ShapeBlend variants used during training, run:

```bash
for i in {1..9}; do
    python blend.py \
        --input-dir ../imagenet1000/clean/train \
        --contours-dir ../imagenet1000/contours/train \
        --output-dir ../imagenet1000/blend${i}/train \
        --alpha 0.${i}

    python blend.py \
        --input-dir ../imagenet1000/clean/train \
        --contours-dir ../imagenet1000/contours/train \
        --output-dir ../imagenet1000/negative${i}/train \
        --alpha 0.${i} \
        --negative
done
```

The resulting dataset structure is:

```text
imagenet1000/
├── clean/
│   ├── train/
│   └── val/
├── contours/train/
├── blend1/train/
├── ...
├── blend9/train/
├── negative1/train/
├── ...
└── negative9/train/
```

### DeepAugment

Following the original DeepAugment methodology, EDSR
and CAE augmented images are generated offline.

When combining ShapeBlend with DeepAugment, ShapeBlend
is applied first. The generated images must follow
the same directory structure:

```text
imagenet1000/
├── clean/
├── blend1/
├── ...
├── negative9/
├── EDSR/
│   ├── clean/train/
│   ├── blend1/train/
│   ├── ...
│   └── negative9/train/
└── CAE/
    ├── clean/train/
    ├── blend1/train/
    ├── ...
    └── negative9/train/
```

### Network Architecture

Both training scripts support ResNet-50 and ConvNeXt-Tiny
by manually selecting the corresponding model definition.

The scripts are currently configured for ConvNeXt-Tiny:

```python
model = convnext_tiny(weights=None)
state_dict = torch.load("weights1000/tiny.pth")
model.load_state_dict(state_dict)
```

To use ResNet-50, uncomment the ResNet-50 definition
and comment out the ConvNeXt-Tiny initialization block.

Set the initial learning rate according to the architecture:

- ResNet-50: `0.1`
- ConvNeXt-Tiny: `0.001`

### Training Commands

Define the common training parameters:

```bash
COMMON_ARGS="--nclass 1000 --epochs 30 --batch 1024 \
--lr 0.1 --wd 0.0001 --m 0.9 --num-gpus 4"
```

The following commands cover all main training
configurations reported in the paper:

```bash
# Baseline
python train.py $COMMON_ARGS --exp baseline --num 1

# ShapeBlend
python train.py $COMMON_ARGS --exp SB --num 1 --online

# AugMix
python train.py $COMMON_ARGS --exp AM --num 1 --augmix

# ShapeBlend + AugMix
python train.py $COMMON_ARGS --exp SB_AM --num 1 --online --augmix

# PRIME
python train_with_prime.py $COMMON_ARGS --exp PR --num 1 --prime

# ShapeBlend + PRIME
python train_with_prime.py $COMMON_ARGS --exp SB_PR --num 1 --online --prime

# DeepAugment
python train.py $COMMON_ARGS --exp DA --num 1 --deepaugment

# ShapeBlend + DeepAugment
python train.py $COMMON_ARGS --exp SB_DA --num 1 --online --deepaugment

# DeepAugment + AugMix
python train.py $COMMON_ARGS --exp DA_AM --num 1 --deepaugment --augmix

# ShapeBlend + DeepAugment + AugMix
python train.py $COMMON_ARGS --exp SB_DA_AM --num 1 --online --deepaugment --augmix

# DeepAugment + PRIME
python train_with_prime.py $COMMON_ARGS --exp DA_PR --num 1 --deepaugment --prime

# ShapeBlend + DeepAugment + PRIME
python train_with_prime.py $COMMON_ARGS --exp SB_DA_PR --num 1 --online --deepaugment --prime
```

Training logs and checkpoints are saved in `train1000/`
and `weights1000/`, respectively.

---

## Evaluation

**Instructions to be added.**


---

## Citation

If you find this work useful, please cite:

```bibtex
@article{azzopardi2026shapeblend,
  title={ShapeBlend: Boosting out-of-distribution robustness in image classification via shape-based blending augmentation},
  author={Azzopardi, George and Esposito, Sabatino and Greco, Antonio and Vento, Mario},
  journal={Computer Vision and Image Understanding},
  publisher={Elsevier}
  volume={268},
  pages={104768},
  year={2026},
  doi={10.1016/j.cviu.2026.104768}
}
