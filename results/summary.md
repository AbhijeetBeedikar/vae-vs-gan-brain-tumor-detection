### Residual: `positive` (main setting)

| Model | Dice (subject mean) | Dice (global) | Pixel AUROC | Pixel AUPRC | Slice AUROC (MSD) | Slice AUROC (vs IXI) | Healthy pixel FPR |
|---|---|---|---|---|---|---|---|
| Intensity baseline | 0.394 ± 0.243 | 0.431 | 0.854 | 0.351 | 0.779 | 0.772 | 4.52% |
| AE | 0.380 ± 0.161 | 0.420 | 0.796 | 0.402 | 0.869 | 0.865 | 4.01% |
| VAE | 0.514 ± 0.227 | 0.566 | 0.894 | 0.575 | 0.867 | 0.929 | 1.21% |
| f-AnoGAN | 0.474 ± 0.183 | 0.517 | 0.855 | 0.514 | 0.867 | 0.931 | 1.31% |

### Residual: `abs`

| Model | Dice (subject mean) | Dice (global) | Pixel AUROC | Pixel AUPRC | Slice AUROC (MSD) | Slice AUROC (vs IXI) | Healthy pixel FPR |
|---|---|---|---|---|---|---|---|
| Intensity baseline | 0.394 ± 0.243 | 0.431 | 0.854 | 0.351 | 0.779 | 0.772 | 4.52% |
| AE | 0.334 ± 0.157 | 0.368 | 0.801 | 0.334 | 0.847 | 0.799 | 6.89% |
| VAE | 0.484 ± 0.222 | 0.540 | 0.880 | 0.542 | 0.866 | 0.917 | 1.61% |
| f-AnoGAN | 0.435 ± 0.187 | 0.483 | 0.864 | 0.477 | 0.861 | 0.903 | 2.62% |
