# Reliable Abnormality Detection in Musculoskeletal X-rays Using Calibration and Uncertainty Estimation

This project studies reliability in deep learning models for musculoskeletal X-ray abnormality detection using the MURA v1.1 dataset.

Instead of only optimizing classification accuracy, the project asks whether X-ray classifiers produce confidence scores that remain meaningful under clean validation data, study-level aggregation, image corruptions, and test-time augmentation.

The main goal is to evaluate not just whether a model is correct, but whether it knows when it may be wrong.

## Research Question

Can convolutional neural networks classify normal versus abnormal musculoskeletal X-rays while producing reliable uncertainty and confidence estimates?

More specifically, this project evaluates:

- Image-level classification performance
- Calibration before and after temperature scaling
- Study-level aggregation across multiple X-ray views
- Study-level temperature scaling
- Robustness under synthetic image corruptions
- Threshold tradeoffs between sensitivity and specificity
- Test-time augmentation uncertainty
- High-confidence failure cases

## Dataset

This project uses the MURA v1.1 musculoskeletal X-ray dataset.

MURA contains upper-extremity X-ray studies labeled as normal or abnormal across multiple body parts, including wrist, shoulder, hand, finger, elbow, forearm, and humerus.

This repository does not include the MURA image files. Users must obtain access separately and place the dataset locally at:

```text
data/raw/MURA-v1.1/
```

The processed manifest used in this project contains:

```text
40,005 total images
36,808 training images
3,197 validation images
1,199 validation studies
7 body parts
2 classes: normal and abnormal
```

## Methods

Two ImageNet-pretrained CNN backbones were fine-tuned for binary abnormality classification:

```text
ResNet18
ResNet50
```

Each model outputs a single abnormality logit. The sigmoid probability is interpreted as the model’s estimated probability that an image is abnormal.

The project evaluates reliability using:

```text
ECE: Expected Calibration Error
Brier score
Negative log-likelihood
Overconfidence gap
Temperature scaling
Study-level aggregation
Test-time augmentation uncertainty
Corruption stress tests
Threshold analysis
Failure-case visualization
```

## Image-Level Results

Validation performance was first evaluated at the image level using the 3,197 MURA validation images.

| Model | Accuracy | AUROC | Precision | Sensitivity | Specificity | F1 | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 | 0.8001 | 0.8688 | 0.8383 | 0.7216 | 0.8722 | 0.7756 | 0.0202 |
| ResNet50 | **0.8151** | **0.8741** | **0.8922** | 0.6980 | **0.9226** | **0.7833** | 0.0370 |

ResNet50 achieved stronger overall classification performance, with higher accuracy, AUROC, precision, specificity, and F1. However, it had worse calibration than ResNet18 before post-hoc calibration.

## Temperature Scaling

Temperature scaling was applied as a post-hoc calibration method. A learned temperature greater than 1 softens the predicted probabilities, reducing overconfidence.

| Model | Temperature | ECE Before | ECE After | Brier Before | Brier After | NLL Before | NLL After |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 | 1.1216 | 0.0202 | 0.0134 | 0.1419 | 0.1420 | 0.4462 | 0.4444 |
| ResNet50 | 1.2261 | 0.0370 | **0.0094** | 0.1373 | **0.1366** | 0.4437 | **0.4369** |

Temperature scaling substantially improved calibration, especially for ResNet50.

The stronger ResNet50 model was more overconfident before calibration, but after temperature scaling it achieved the best overall calibrated image-level performance.

## Study-Level Aggregation

MURA labels are study-level labels, while each study may contain multiple images. To better align evaluation with the dataset structure, image-level probabilities were aggregated into study-level predictions.

Three aggregation methods were tested:

```text
mean: average probability across all images in a study
max: maximum abnormal probability across images
top2_mean: average of the two highest abnormal probabilities
```

### ResNet50 Study-Level Results

| Aggregation | Accuracy | AUROC | Sensitivity | Specificity | F1 | ECE |
|---|---:|---:|---:|---:|---:|---:|
| mean | 0.8232 | **0.8819** | 0.6840 | **0.9365** | 0.7764 | **0.0156** |
| max | 0.8207 | 0.8806 | **0.7584** | 0.8714 | **0.7915** | 0.0266 |
| top2_mean | **0.8249** | 0.8809 | 0.7119 | 0.9168 | 0.7848 | 0.0208 |

Study-level aggregation improved performance relative to image-level evaluation. The aggregation rule controlled the clinical tradeoff: mean aggregation was more specific and better calibrated, while max aggregation improved sensitivity and F1.

## Study-Level Temperature Scaling

Temperature scaling was also learned directly at the study level by optimizing the final aggregated study probabilities.

| Model | Aggregation | Study Temperature | Vanilla NLL | Scaled NLL | Vanilla ECE | Scaled ECE |
|---|---|---:|---:|---:|---:|---:|
| ResNet18 | mean | 1.00 | 0.4280 | 0.4280 | 0.0267 | 0.0267 |
| ResNet18 | max | 1.18 | 0.4558 | 0.4525 | 0.0213 | 0.0290 |
| ResNet18 | top2_mean | 1.04 | 0.4315 | 0.4314 | 0.0193 | 0.0273 |
| ResNet50 | mean | 1.14 | 0.4274 | 0.4250 | **0.0156** | 0.0191 |
| ResNet50 | max | 1.17 | 0.4329 | 0.4292 | 0.0266 | **0.0188** |
| ResNet50 | top2_mean | 1.12 | 0.4225 | **0.4207** | 0.0208 | 0.0200 |

Study-level temperature scaling improved negative log-likelihood across ResNet50 aggregation methods. ECE improvements depended on the aggregation rule, suggesting that calibration should be evaluated at the same prediction level where the model is deployed.

## Test-Time Augmentation Uncertainty

Test-time augmentation was used as a lightweight uncertainty estimation method. Each validation image was evaluated under eight mild deterministic perturbations:

```text
original image
brightness down
brightness up
contrast down
contrast up
rotation -5 degrees
rotation +5 degrees
slight center crop
```

The mean probability was used for classification. Prediction variance, entropy, and margin uncertainty were used as uncertainty signals.

| Model | Accuracy | AUROC | F1 | ECE | Entropy Error-Detection AUROC | Margin Error-Detection AUROC |
|---|---:|---:|---:|---:|---:|---:|
| ResNet18 + TTA | 0.8017 | 0.8708 | 0.7835 | **0.0153** | **0.7375** | **0.7375** |
| ResNet50 + TTA | **0.8183** | **0.8745** | **0.7894** | 0.0191 | 0.7167 | 0.7167 |

Incorrect predictions had higher uncertainty than correct predictions.

For ResNet50:

```text
Correct prediction entropy:    0.3630
Incorrect prediction entropy:  0.5076
```

For ResNet18:

```text
Correct prediction entropy:    0.3967
Incorrect prediction entropy:  0.5608
```

This suggests that TTA uncertainty can help flag unreliable predictions, although it does not eliminate all high-confidence failures.

## Robustness Under Image Corruptions

Models were evaluated under synthetic image corruptions, including Gaussian noise, blur, brightness shifts, and contrast shifts.

Selected ResNet50 results:

| Condition | Severity | Accuracy | AUROC | F1 | ECE | Mean Confidence |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0 | 0.8151 | 0.8741 | 0.7833 | 0.0370 | 0.8425 |
| gaussian_noise | 4 | 0.5317 | 0.4456 | 0.0507 | 0.1798 | 0.7116 |
| blur | 4 | 0.6325 | 0.6997 | 0.6154 | 0.0423 | 0.6748 |
| brightness_up | 4 | 0.7792 | 0.8377 | 0.7328 | 0.0497 | 0.8288 |
| contrast_up | 4 | 0.7911 | 0.8458 | 0.7539 | 0.0563 | 0.8414 |

Severe Gaussian noise caused the largest reliability failure: performance dropped sharply while confidence remained high. This shows that good clean calibration does not guarantee robustness under distribution shift.

## Threshold Analysis

The default 0.5 threshold was not always the best operating point.

For ResNet50, lowering the threshold improved F1 and sensitivity:

| Probability Type | Operating Point | Threshold | Accuracy | Sensitivity | Specificity | F1 |
|---|---|---:|---:|---:|---:|---:|
| vanilla | default | 0.50 | 0.8151 | 0.6980 | 0.9226 | 0.7833 |
| vanilla | best F1 | 0.30 | 0.8167 | 0.8007 | 0.8314 | **0.8070** |
| temperature scaled | best F1 | 0.35 | **0.8183** | 0.7895 | 0.8446 | 0.8061 |

This shows that ResNet50’s lower sensitivity at the default threshold was partly an operating-point issue rather than only a model-capacity issue.

## Failure-Case Analysis

High-confidence failure cases were visualized for the calibrated ResNet50 model.

The most important failure type was high-confidence false negatives: abnormal studies predicted as normal with high confidence. These cases are clinically important because they show that calibration and uncertainty methods improve aggregate reliability but do not guarantee safety on every individual prediction.

Example figure paths:

```text
assets/figures/resnet50_reliability_diagram.png
assets/figures/resnet50_temp_scaled_ece_under_corruption.png
assets/figures/resnet50_threshold_sensitivity_specificity.png
assets/figures/resnet50_confident_false_negatives.png
assets/figures/resnet50_uncertain_wrong.png
```

## Key Findings

1. ResNet50 achieved the strongest image-level classification performance, but it was less calibrated than ResNet18 before temperature scaling.

2. Temperature scaling substantially improved image-level calibration, especially for ResNet50.

3. Study-level aggregation improved performance relative to image-level evaluation and better matched the structure of the MURA dataset.

4. The aggregation rule controlled sensitivity-specificity tradeoffs: mean aggregation was more specific, max aggregation was more sensitive, and top2_mean provided a balanced alternative.

5. Study-level temperature scaling improved ResNet50 negative log-likelihood across aggregation methods, but ECE improvements depended on the aggregation rule.

6. Test-time augmentation provided a useful uncertainty signal. Incorrect predictions had higher entropy and margin uncertainty than correct predictions.

7. Severe image corruptions revealed reliability failures that were not visible from clean validation metrics alone.

8. High-confidence false negatives remained, showing that calibration and uncertainty estimation are useful risk signals but not safety guarantees.

## Repository Structure

```text
mura-uncertainty-calibration/
  README.md
  requirements.txt

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
      study_level_aggregation.py
      study_level_temperature_scaling.py
      tta_uncertainty.py

    visualization/
      failure_cases.py

  results/
    model_comparison.csv
    robustness_summary.csv
    threshold_summary.csv
    study_level_summary_vanilla.csv
    study_level_summary_temperature_scaled.csv
    study_level_temperature_scaling_summary.csv
    tta_uncertainty_summary.csv

  reports/
    final_analysis.md

  assets/
    figures/
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Expected local data layout:

```text
data/raw/MURA-v1.1/
data/manifests/mura_manifest.csv
```

Build or inspect the manifest:

```bash
python -m src.data.build_manifest
python -m src.data.inspect_manifest
```

Train a baseline model:

```bash
python -m src.models.train_baseline --backbone resnet18
python -m src.models.train_baseline --backbone resnet50
```

Evaluate a trained model:

```bash
python -m src.evaluation.evaluate_baseline --backbone resnet50
```

Run calibration analysis:

```bash
python -m src.evaluation.calibration_analysis --backbone resnet50
python -m src.evaluation.temperature_scaling --backbone resnet50
```

Run robustness tests:

```bash
python -m src.evaluation.stress_test_corruptions --backbone resnet50
python -m src.evaluation.stress_test_temperature_scaled --backbone resnet50
```

Run study-level evaluation:

```bash
python -m src.evaluation.study_level_aggregation
python -m src.evaluation.study_level_aggregation --use-temp-scaled
```

Run study-level temperature scaling:

```bash
python -m src.evaluation.study_level_temperature_scaling --backbone resnet50 --aggregation mean
python -m src.evaluation.study_level_temperature_scaling --backbone resnet50 --aggregation max
python -m src.evaluation.study_level_temperature_scaling --backbone resnet50 --aggregation top2_mean
```

Run TTA uncertainty:

```bash
python -m src.evaluation.tta_uncertainty \
  --backbone resnet50 \
  --checkpoint-path outputs/checkpoints/baseline_resnet50_best.pt \
  --batch-size 16
```

## Limitations

This project is a research-style reliability analysis, not a clinical diagnostic system.

Important limitations:

- MURA labels are study-level, while some experiments use image-level evaluation.
- Evaluation is performed on the MURA validation set, not an external hospital dataset.
- Synthetic corruptions do not fully represent real clinical distribution shifts.
- The models are standard CNN baselines, not specialized radiology architectures.
- Temperature scaling improves aggregate calibration but does not eliminate individual high-confidence errors.
- TTA uncertainty helps flag risky predictions but is not a guarantee of correctness.
- No clinician review was performed for failure cases.

## Future Work

Potential extensions include:

- External validation on another musculoskeletal X-ray dataset
- Multi-seed training for stronger statistical confidence
- Higher-resolution training
- DenseNet or EfficientNet baselines
- Study-level calibration with a held-out calibration split
- MC dropout or deep ensembles
- Body-part-specific calibration analysis
- Clinical review of high-confidence false negatives

## Conclusion

This project shows that musculoskeletal X-ray classifiers should be evaluated beyond accuracy alone.

ResNet50 achieved the best classification performance, but reliability analysis revealed important differences between clean accuracy, calibration, robustness, threshold behavior, and uncertainty. Temperature scaling and TTA improved reliability, while study-level aggregation better matched the structure of the MURA dataset.

The main takeaway is that confidence-aware evaluation can expose model behavior that ordinary accuracy metrics miss.