# Reliable MURA X-ray

Uncertainty-aware abnormality detection in musculoskeletal X-rays using calibration, threshold analysis, and corruption robustness testing.

This project studies whether deep learning models for medical image classification can produce confidence scores that are actually useful. Using the MURA v1.1 musculoskeletal X-ray dataset, I trained ResNet-based abnormality classifiers and evaluated not only clean validation performance, but also calibration, temperature scaling, robustness under image corruptions, threshold tradeoffs, and high-confidence failure cases.

The goal is not just to ask:

> Can the model classify X-rays as normal or abnormal?

but also:

> Does the model know when it might be wrong?

---

## Research Question

Can musculoskeletal X-ray classifiers produce confidence scores that remain reliable under clean and shifted image conditions?

More specifically, this project evaluates:

- binary abnormality detection on MURA X-rays
- calibration of predicted probabilities
- post-hoc temperature scaling
- robustness under synthetic image corruptions
- sensitivity/specificity tradeoffs across decision thresholds
- high-confidence false positive and false negative failure cases

---

## Dataset

This project uses the MURA v1.1 musculoskeletal X-ray dataset, which contains upper-extremity X-ray studies labeled as normal or abnormal.

After preprocessing and removing invalid AppleDouble metadata files, the cleaned dataset contained:

| Split | Images |
|---|---:|
| Train | 36,808 |
| Validation | 3,197 |
| Total | 40,005 |

The dataset includes seven body-part categories:

- `XR_WRIST`
- `XR_SHOULDER`
- `XR_HAND`
- `XR_FINGER`
- `XR_ELBOW`
- `XR_FOREARM`
- `XR_HUMERUS`

Labels were parsed from study folder names:

| Folder pattern | Label |
|---|---|
| `study*_negative` | normal |
| `study*_positive` | abnormal |

A manifest file was generated with image path, split, body part, patient ID, study ID, label, and label name.

---

## Methods

### Models

Two ImageNet-pretrained ResNet backbones were fine-tuned for binary abnormality classification:

| Model | Input size | Output |
|---|---|---|
| ResNet18 | 224 × 224 RGB | single abnormality logit |
| ResNet50 | 224 × 224 RGB | single abnormality logit |

Although MURA images are grayscale, they were converted to 3-channel RGB to use standard ImageNet-pretrained torchvision models.

### Evaluation

The project evaluates each model using:

- accuracy
- AUROC
- precision
- recall / sensitivity
- specificity
- F1 score
- confusion matrix
- expected calibration error, 10 bins
- Brier score
- negative log likelihood
- mean confidence
- overconfidence gap

### Calibration

Temperature scaling was applied as a post-hoc calibration method using clean validation logits. Temperature scaling changes probability confidence but does not change the ranking of predictions, so accuracy, AUROC, and F1 at threshold 0.5 remain unchanged.

### Robustness Stress Testing

Validation images were corrupted using controlled synthetic transformations:

- Gaussian noise
- Gaussian blur
- brightness decrease
- brightness increase
- contrast decrease
- contrast increase

Each corruption was evaluated at severity levels 1 through 4. The purpose was to test whether model confidence decreases when image quality degrades.

### Threshold Analysis

The default threshold of 0.5 was compared against thresholds from 0.05 to 0.95. This was used to study the clinical tradeoff between sensitivity and specificity.

### Failure-Case Visualization

Representative prediction cases were visualized, including:

- confident correct normal predictions
- confident correct abnormal predictions
- uncertain wrong predictions
- high-confidence false negatives
- high-confidence false positives

---

## Clean Validation Results

### Model Comparison

| Model | Accuracy | AUROC | Precision | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|
| ResNet18 | 0.8001 | 0.8688 | 0.8383 | 0.7216 | 0.8722 | 0.7756 |
| ResNet50 | **0.8151** | **0.8741** | **0.8922** | 0.6980 | **0.9226** | **0.7833** |

ResNet50 improved clean accuracy, AUROC, precision, specificity, and F1 compared to ResNet18. However, at the default 0.5 threshold, ResNet50 had lower sensitivity, meaning it missed more abnormal cases.

### Confusion Matrices

#### ResNet18

|  | Predicted normal | Predicted abnormal |
|---|---:|---:|
| True normal | 1454 | 213 |
| True abnormal | 426 | 1104 |

#### ResNet50

|  | Predicted normal | Predicted abnormal |
|---|---:|---:|
| True normal | 1538 | 129 |
| True abnormal | 462 | 1068 |

At threshold 0.5, ResNet50 was more conservative: it produced fewer false positives but more false negatives.

---

## Calibration Results

### Before Temperature Scaling

| Model | ECE | Brier score | NLL | Mean confidence | Overconfidence gap |
|---|---:|---:|---:|---:|---:|
| ResNet18 | **0.0202** | 0.1419 | 0.4462 | 0.8101 | +0.0100 |
| ResNet50 | 0.0370 | **0.1373** | **0.4437** | 0.8425 | +0.0274 |

ResNet50 had better Brier score and NLL, but worse ECE and a larger overconfidence gap. This suggests that ResNet50 was more accurate overall, but also more overconfident before calibration.

### Temperature Scaling

| Model | Temperature | ECE before | ECE after | NLL before | NLL after |
|---|---:|---:|---:|---:|---:|
| ResNet18 | 1.1216 | 0.0202 | 0.0134 | 0.4462 | 0.4444 |
| ResNet50 | 1.2261 | 0.0370 | **0.0094** | 0.4437 | **0.4369** |

Both learned temperatures were greater than 1, indicating mild overconfidence. ResNet50 required stronger softening, but after temperature scaling it achieved the best clean calibration.

Temperature scaling preserved classification performance while improving probability calibration:

| Model | Accuracy | AUROC | F1 | Temp-scaled ECE | Temp-scaled overconfidence gap |
|---|---:|---:|---:|---:|---:|
| ResNet18 | 0.8001 | 0.8688 | 0.7756 | 0.0134 | -0.0074 |
| ResNet50 | **0.8151** | **0.8741** | **0.7833** | **0.0094** | -0.0063 |

---

## Robustness Under Image Corruptions

The models were evaluated under corrupted validation images. The table below shows selected severe corruption conditions.

| Model | Corruption | Severity | Accuracy | AUROC | ECE | Mean confidence | Temp-scaled ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| ResNet18 | clean | 0 | 0.8001 | 0.8688 | 0.0202 | 0.8101 | 0.0130 |
| ResNet18 | Gaussian noise | 4 | 0.5427 | 0.5603 | 0.0731 | 0.6158 | 0.0605 |
| ResNet18 | blur | 4 | 0.6647 | 0.7294 | 0.0405 | 0.7051 | 0.0241 |
| ResNet18 | brightness up | 4 | 0.7582 | 0.8315 | 0.0427 | 0.8009 | 0.0268 |
| ResNet18 | contrast up | 4 | 0.7748 | 0.8427 | 0.0444 | 0.8171 | 0.0270 |
| ResNet50 | clean | 0 | 0.8151 | 0.8741 | 0.0370 | 0.8425 | 0.0094 |
| ResNet50 | Gaussian noise | 4 | 0.5317 | 0.4456 | 0.1798 | 0.7116 | 0.1448 |
| ResNet50 | blur | 4 | 0.6325 | 0.6997 | 0.0423 | 0.6748 | 0.0359 |
| ResNet50 | brightness up | 4 | 0.7792 | 0.8377 | 0.0497 | 0.8288 | 0.0241 |
| ResNet50 | contrast up | 4 | 0.7911 | 0.8458 | 0.0563 | 0.8414 | 0.0254 |

### Robustness Findings

Temperature scaling substantially improved calibration on clean data and reduced overconfidence under brightness and contrast shifts.

For ResNet50:

| Condition | Vanilla ECE | Temp-scaled ECE |
|---|---:|---:|
| Clean | 0.0370 | 0.0094 |
| Brightness up, severity 4 | 0.0497 | 0.0241 |
| Contrast up, severity 4 | 0.0563 | 0.0254 |
| Gaussian noise, severity 4 | 0.1786 | 0.1448 |

The most important failure mode was severe Gaussian noise. Under Gaussian noise severity 4, ResNet50 reached an AUROC below 0.5 and remained substantially miscalibrated even after temperature scaling.

This suggests that post-hoc calibration helps, but does not guarantee reliable confidence under severe distribution shift.

---

## Threshold Analysis

At the default threshold of 0.5, ResNet50 had higher specificity but lower sensitivity than ResNet18. Threshold analysis showed that this was partly an operating-point issue.

### Best-F1 Operating Points

| Model | Probability type | Threshold | Accuracy | Sensitivity | Specificity | F1 |
|---|---|---:|---:|---:|---:|---:|
| ResNet18 | vanilla | 0.40 | 0.7939 | 0.7889 | 0.7984 | 0.7856 |
| ResNet18 | temperature scaled | 0.45 | 0.8011 | 0.7634 | 0.8356 | 0.7860 |
| ResNet50 | vanilla | 0.30 | 0.8167 | **0.8007** | 0.8314 | **0.8070** |
| ResNet50 | temperature scaled | 0.35 | **0.8183** | 0.7895 | **0.8446** | 0.8061 |

After threshold tuning, ResNet50 achieved the best F1 score and a better sensitivity-specificity balance.

### High-Sensitivity Operating Points

For a screening-style setting with sensitivity greater than or equal to 0.90:

| Model | Probability type | Threshold | Sensitivity | Specificity | False positives | False negatives |
|---|---|---:|---:|---:|---:|---:|
| ResNet18 | vanilla | 0.20 | 0.9065 | 0.5153 | 808 | 143 |
| ResNet18 | temperature scaled | 0.20 | 0.9242 | 0.4607 | 899 | 116 |
| ResNet50 | vanilla | 0.10 | **0.9438** | 0.3641 | 1060 | **86** |
| ResNet50 | temperature scaled | 0.15 | 0.9346 | 0.3911 | 1015 | 100 |

Lowering the threshold reduces missed abnormalities but substantially increases false positives. This reflects a realistic clinical tradeoff between screening sensitivity and unnecessary follow-up.

---

## Failure-Case Analysis

Failure-case visualization showed that calibration improvements at the aggregate level do not eliminate individual confident mistakes.

The most clinically important examples were high-confidence false negatives: abnormal images predicted as normal with confidence above 0.93 after temperature scaling.

These cases highlight a key limitation:

> A model can be well-calibrated on average while still producing individual high-confidence errors.

Uncertain wrong cases were also observed, where predicted probabilities were close to 0.5. These are less concerning because the model does not strongly commit to an incorrect label.

---

## Key Findings

1. ResNet50 slightly improved clean classification performance over ResNet18.
2. ResNet50 was more overconfident before calibration.
3. Temperature scaling improved clean calibration, especially for ResNet50.
4. Temperature scaling reduced overconfidence under brightness and contrast shifts.
5. Severe Gaussian noise remained a major unresolved failure mode.
6. Threshold tuning showed that ResNet50’s lower sensitivity at threshold 0.5 was partly an operating-point issue.
7. High-confidence false negatives remained even after temperature scaling.

---

## Repository Structure

```text
reliable-mura-xray/
  data/
    raw/                  # ignored; contains MURA images locally
    manifests/            # generated manifest CSV
  outputs/
    evaluation/           # metrics, predictions, figures
    reports/              # generated comparison summaries
  reports/
    final_analysis.md     # paper-style writeup
  src/
    data/
      build_manifest.py
      inspect_manifest.py
      visualize_samples.py
      mura_dataset.py
      check_dataloader.py
    models/
      train_baseline.py
    evaluation/
      evaluate_baseline.py
      calibration_analysis.py
      temperature_scaling.py
      stress_test_corruptions.py
      stress_test_temperature_scaled.py
      compare_models.py
      summarize_robustness.py
      threshold_analysis.py
      summarize_thresholds.py
    visualization/
      failure_cases.py
```

---

## How to Run

### 1. Build the manifest

```bash
python -m src.data.build_manifest
```

### 2. Inspect the dataset

```bash
python -m src.data.inspect_manifest
python -m src.data.visualize_samples
python -m src.data.check_dataloader
```

### 3. Train a baseline model

Set the backbone in `src/models/train_baseline.py`, then run:

```bash
python -m src.models.train_baseline
```

### 4. Evaluate a trained model

```bash
python -m src.evaluation.evaluate_baseline --backbone resnet18
python -m src.evaluation.evaluate_baseline --backbone resnet50
```

### 5. Run calibration analysis

```bash
python -m src.evaluation.calibration_analysis --backbone resnet18
python -m src.evaluation.calibration_analysis --backbone resnet50
```

### 6. Run temperature scaling

```bash
python -m src.evaluation.temperature_scaling --backbone resnet18
python -m src.evaluation.temperature_scaling --backbone resnet50
```

### 7. Run corruption stress tests

```bash
python -m src.evaluation.stress_test_corruptions --backbone resnet18
python -m src.evaluation.stress_test_corruptions --backbone resnet50
```

### 8. Run temperature-scaled stress tests

```bash
python -m src.evaluation.stress_test_temperature_scaled --backbone resnet18
python -m src.evaluation.stress_test_temperature_scaled --backbone resnet50
```

### 9. Generate summary reports

```bash
python -m src.evaluation.compare_models
python -m src.evaluation.summarize_robustness
python -m src.evaluation.summarize_thresholds
```

### 10. Generate failure-case visualizations

```bash
python -m src.visualization.failure_cases --backbone resnet50 --use-temp-scaled
```

---

## Limitations

This project is an experimental research-style analysis, not a clinical system.

Current limitations include:

- Evaluation is primarily image-level, while MURA labels are study-level.
- No external validation dataset was used.
- Synthetic corruptions are controlled stress tests, not perfect substitutes for real scanner or hospital domain shift.
- The models are standard CNN baselines rather than medical-domain foundation models.
- Temperature scaling was learned on clean validation data and may not generalize to all shifted conditions.
- High-confidence false negatives remain possible even after calibration.

---

## Next Steps

Potential extensions include:

- study-level aggregation of image predictions
- additional backbones such as ResNet152, DenseNet121, and EfficientNet
- higher-resolution training
- multi-seed robustness evaluation
- test-time augmentation uncertainty
- Monte Carlo dropout uncertainty
- external validation on another musculoskeletal X-ray dataset
- cleaner public repo release

---

## Project Goal

This project evaluates abnormality detection through the lens of trustworthiness rather than accuracy alone. In medical AI, a model that is confidently wrong can be more concerning than a model that is uncertain. The central goal is to understand when model confidence is useful, when it fails, and how simple calibration methods such as temperature scaling can help.