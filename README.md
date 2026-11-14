# Assignment 2 — Rice Leaf Disease Classification

Advanced Machine Learning · Nayebul Jannath Chowdhury

This project compares four classifiers under **Method 1 (Dataset-Based Evaluation)**. Each classifier was trained and evaluated separately on D1 and D2. All models used the same frozen training, validation, and test partitions within each dataset.

[Technical report (Word)](./Assignment02_22646206_Technical_Report.docx)

## Datasets

| Dataset | Source | Classes | Train | Validation | Test |
| --- | --- | ---: | ---: | ---: | ---: |
| D1 | [Rice Leaf Bacterial and Fungal Disease Dataset, v2](https://data.mendeley.com/datasets/hx6f852hw4/2) | 8 | 1,156 | 247 | 248 |
| D2 | [Rice Leaf Disease Image Samples, v2](https://data.mendeley.com/datasets/fwcj7stb8r/2) | 4 | 3,327 | 712 | 713 |

Frozen splits: [D1 manifest](./Assignment02_Verified_Splits/D1_split_manifest.csv) · [D2 manifest](./Assignment02_Verified_Splits/D2_split_manifest.csv) · [Class maps](./Assignment02_Verified_Splits/class_maps.json)

## Notebooks and evidence

| Model | D1 | D2 |
| --- | --- | --- |
| MFA-SqueezeNet | [Notebook](./notebooks/Experiment-1_D1/assignment02-paper6-mfa-squeezenet-d1-revised.ipynb) · [Evidence](./notebooks/Experiment-1_D1) | [Notebook](./notebooks/Experiment-1_D2/assignment02-paper6-mfa-squeezenet-d2-revised.ipynb) · [Evidence](./notebooks/Experiment-1_D2) |
| MobileNetV3-Large | [Notebook](./notebooks/Experiment-2_D1/assignment02-experiment3-d1.ipynb) · [Evidence](./notebooks/Experiment-2_D1) | [Notebook](./notebooks/Experiment-2_D2/assignment02-experiment3-d2.ipynb) · [Evidence](./notebooks/Experiment-2_D2) |
| AMFF-Hybrid Net | [Notebook](./notebooks/Experiment-3_D1/assignment02-paper5-amff-d1.ipynb) · [Evidence](./notebooks/Experiment-3_D1) | [Notebook](./notebooks/Experiment-3_D2/assignment02-paper5-amff-d2.ipynb) · [Evidence](./notebooks/Experiment-3_D2) |
| Twin CNN | [Notebook](./notebooks/Experiment-4_D1/assignment-02-paper-9-twin-cnn-d1.ipynb) · [Evidence](./notebooks/Experiment-4_D1) | [Notebook](./notebooks/Experiment-4_D2/assignment-02-paper-9-twin-cnn-d2.ipynb) · [Evidence](./notebooks/Experiment-4_D2) |

Each experiment folder contains the notebook and its available configuration, logs, predictions, and figures.

## Models and test results

| Model | Paper | D1 accuracy | D1 macro F1 | D2 accuracy | D2 macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| MFA-SqueezeNet | [Journal of Phytopathology, 2025](https://doi.org/10.1111/jph.70092) | **69.76%** | 0.6482 | **99.02%** | **0.9897** |
| MobileNetV3-Large baseline | [Journal of Applied Informatics and Computing, 2026](https://doi.org/10.30871/jaic.v10i1.12033) | 69.35% | **0.6706** | 93.13% | 0.9284 |
| AMFF-Hybrid Net, six epochs with Otsu | [Scientific Reports, 2026](https://doi.org/10.1038/s41598-026-52534-3) | 44.35% | 0.4139 | 83.87% | 0.8284 |
| Twin CNN | [Journal of Big Data, 2025](https://doi.org/10.1186/s40537-025-01148-z) | 63.31% | 0.6147 | 94.39% | 0.9412 |

### D1 AMFF-Hybrid Net extension

The two 20-epoch runs used the same 247 D1 validation images. The [extension notebook](./notebooks/Experiment-3_D1/Extended%20Experiment%20Version/assignment02-paper5-amff-d1.ipynb) reproduces both variants. The [extension folder](./notebooks/Experiment-3_D1/Extended%20Experiment%20Version) contains the validation summaries and epoch logs.

| Preprocessing | Validation accuracy | Validation macro F1 |
| --- | ---: | ---: |
| Otsu masking | 63.97% (158/247) | 0.6122 |
| Contrast stretch only | 66.80% (165/247) | 0.6464 |

## Reproduce on Kaggle

1. Import the notebook for your model and dataset into Kaggle.
2. Attach the matching image dataset and frozen split input. The notebooks use `assignment02-d1-original-rice-leaf-1701` for D1, `assignment02-d2-rice-leaf-5932` for D2, and `assignment02-verified-train-val-test-splits` for the manifests. If you mount your own copies, match the notebook's expected paths.
3. Select a T4 GPU and allow access to ImageNet weights where required. Run all cells in order and compare the exported metrics with the experiment evidence.
4. For the D1 AMFF-Hybrid Net extension, run the initialization cells first to generate the shared starting weights. Compare the Otsu and stretch-only runs on the D1 validation split.
