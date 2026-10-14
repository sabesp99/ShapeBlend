# ShapeBlend

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

![ShapeBlend pipeline](assets/pipeline.png)

---

## Pretrained Weights

Pretrained ResNet-50 weights are available in the pretrained folder

## Summary of Results

Below is a concise summary of the main results reported in the paper.

| Method | ImageNet Acc. ↑ | ImageNet-C mCA ↑ | ImageNet-R Acc. ↑ |
|-------|------------------|------------------|-------------------|
| Baseline (ResNet-50) | 76.68 | 39.60 | 36.57 |
| ShapeBlend | 76.92 | 43.61 | 38.90 |
| ShapeBlend + SEV_S | 77.07 | 46.11 | 40.70 |
| ShapeBlend + AugMix | 77.16 | 49.76 | 44.12 |
| ShapeBlend + AugMix + SEV_S | 77.01 | 51.35 | 45.56 |
| ShapeBlend + DeepAugment + Prime | 75.09 | 60.75 | 49.17 |
| ShapeBlend + DeepAugment + Prime + SEV_S | 75.28 | 61.11 | 50.60 |

For full experimental details, please refer to the paper.

---

## Code

**Code: TO BE ADDED**

We are currently organizing the training and evaluation pipeline for public release.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{azzopardi2026shapeblend,
  title={ShapeBlend: Boosting out-of-distribution robustness in image classification via shape-based blending augmentation},
  author={Azzopardi, George and Esposito, Sabatino and Greco, Antonio and Vento, Mario},
  journal={Computer Vision and Image Understanding},
  volume={268},
  pages={104768},
  year={2026},
  doi={10.1016/j.cviu.2026.104768}
}