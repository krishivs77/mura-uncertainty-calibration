# Final Analysis: Reliable Abnormality Detection in Musculoskeletal X-rays Using Calibration and Uncertainty Estimation

## Abstract

This project evaluated reliability in deep learning models for musculoskeletal X-ray abnormality detection using the MURA v1.1 dataset. Rather than focusing only on classification accuracy, the analysis studied whether model confidence scores remained meaningful under clean validation data, temperature scaling, study-level aggregation, synthetic image corruptions, threshold changes, and test-time augmentation.

Two ImageNet-pretrained convolutional neural networks, ResNet18 and ResNet50, were fine-tuned for binary normal versus abnormal classification. ResNet50 achieved the strongest image-level classification performance, with validation accuracy of 0.8151 and AUROC of 0.8741. However, it was also more overconfident than ResNet18 before calibration. Temperature scaling reduced ResNet50 image-level ECE from 0.0370 to 0.0094.

Because MURA labels are study-level labels, image predictions were also aggregated into study-level predictions using mean, max, and top-2 mean aggregation. Study-level aggregation improved performance, with ResNet50 reaching study-level accuracy of 0.8249 and AUROC of 0.8819 depending on aggregation method. Test-time augmentation provided an additional uncertainty signal: incorrect predictions had higher entropy and margin uncertainty than correct predictions, with entropy-based error-detection AUROC reaching 0.7375 for ResNet18 and 0.7167 for ResNet50.

Overall, the project shows that accuracy alone is insufficient for evaluating medical imaging classifiers. Calibration, robustness, uncertainty, study-level behavior, and failure-case analysis all reveal important aspects of model reliability.

## 1. Motivation

Medical image classifiers are often evaluated using standard performance metrics such as accuracy, AUROC, sensitivity, specificity, and F1 score. These metrics are important, but they do not fully describe whether a model’s confidence is trustworthy.

For medical AI, a model that is correct 80 percent of the time but overconfident when wrong can be dangerous. A more reliable model should not only make accurate predictions, but also produce uncertainty signals that help identify risky predictions.

This project was motivated by the following question:

> Can convolutional neural networks classify normal versus abnormal musculoskeletal X-rays while producing reliable uncertainty and confidence estimates?

To answer this, the project evaluated model behavior across several reliability dimensions:

```text
image-level classification
image-level calibration
temperature scaling
study-level aggregation
study-level calibration
corruption robustness
threshold tradeoffs
test-time augmentation uncertainty
high-confidence failure cases
```

The goal was not to build a clinical diagnostic system. Instead, the goal was to perform a practical, research-style reliability analysis of deep X-ray classifiers.

## 2. Dataset

The project used the MURA v1.1 musculoskeletal X-ray dataset. MURA contains upper-extremity X-ray studies labeled as normal or abnormal.

The processed dataset contained:

```text
40,005 total images
36,808 training images
3,197 validation images
1,199 validation studies
7 body parts
2 classes: normal and abnormal
```

The seven body parts were:

```text
XR_WRIST
XR_SHOULDER
XR_HAND
XR_FINGER
XR_ELBOW
XR_FOREARM
XR_HUMERUS
```

A key feature of the dataset is that labels are assigned at the study level. A study can contain multiple images or views, and the positive or negative label applies to the study as a whole.

This matters because a simple image-level evaluation does not fully match the dataset structure. For this reason, the project included both image-level evaluation and study-level aggregation.

## 3. Experimental Setup

### 3.1 Models

Two ImageNet-pretrained convolutional neural network backbones were fine-tuned:

```text
ResNet18
ResNet50
```

Each model was modified for binary classification by replacing the final classification layer with a single-logit output. The sigmoid of this logit was interpreted as the predicted probability of abnormality.

### 3.2 Input Processing

MURA images were loaded as grayscale X-rays and converted to RGB to match ImageNet-pretrained backbone input requirements. Images were resized to 224 x 224 and normalized using ImageNet normalization.

### 3.3 Metrics

Classification performance was evaluated using:

```text
accuracy
AUROC
precision
sensitivity / recall
specificity
F1 score
confusion matrix counts
```

Reliability and calibration were evaluated using:

```text
Expected Calibration Error
Brier score
negative log-likelihood
mean confidence
overconfidence gap
```

Uncertainty was evaluated using:

```text
TTA probability standard deviation
entropy
margin uncertainty
error-detection AUROC
```

## 4. Image-Level Classification Results

The first evaluation was performed at the image level using 3,197 validation images.

| Model | Accuracy | AUROC | Precision | Sensitivity | Specificity | F1 | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 | 0.8001 | 0.8688 | 0.8383 | 0.7216 | 0.8722 | 0.7756 | 0.0202 |
| ResNet50 | **0.8151** | **0.8741** | **0.8922** | 0.6980 | **0.9226** | **0.7833** | 0.0370 |

ResNet50 achieved stronger image-level classification performance than ResNet18. It had higher accuracy, AUROC, precision, specificity, and F1 score.

However, ResNet50 had lower sensitivity at the default threshold of 0.5. It produced fewer false positives but more false negatives, meaning it behaved more conservatively than ResNet18.

ResNet50 also had worse calibration before post-hoc calibration, with ECE of 0.0370 compared to ResNet18’s 0.0202. This created an important early finding:

> The stronger classifier was not automatically the more reliable classifier.

This distinction became central to the rest of the analysis.

## 5. Image-Level Calibration and Temperature Scaling

Temperature scaling was used as a post-hoc calibration method. It rescales model logits by a learned scalar temperature before applying the sigmoid function.

A temperature greater than 1 softens predictions and usually indicates that the original model was overconfident.

| Model | Temperature | ECE Before | ECE After | Brier Before | Brier After | NLL Before | NLL After |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 | 1.1216 | 0.0202 | 0.0134 | 0.1419 | 0.1420 | 0.4462 | 0.4444 |
| ResNet50 | 1.2261 | 0.0370 | **0.0094** | 0.1373 | **0.1366** | 0.4437 | **0.4369** |

Both models learned temperatures greater than 1, suggesting that both were overconfident to some extent. ResNet50 required a larger temperature than ResNet18, which matched its higher pre-calibration ECE and overconfidence gap.

Temperature scaling substantially improved ResNet50 calibration:

```text
ECE: 0.0370 → 0.0094
Brier score: 0.1373 → 0.1366
NLL: 0.4437 → 0.4369
```

This means that ResNet50 was the strongest model after calibration, even though it was less calibrated before temperature scaling.

The main calibration finding was:

> Model capacity improved classification performance, but post-hoc calibration was needed to make the stronger model’s confidence more reliable.

## 6. Study-Level Aggregation

Because MURA labels are assigned at the study level, image-level predictions were aggregated into study-level predictions.

Three aggregation rules were tested:

```text
mean: average abnormal probability across all images in the study
max: maximum abnormal probability across images
top2_mean: average of the two highest abnormal probabilities
```

These aggregation rules represent different clinical behaviors. Mean aggregation is smoother and more conservative. Max aggregation treats one highly abnormal image as enough to make the study abnormal. Top-2 mean is a compromise between the two.

### 6.1 ResNet18 Study-Level Results

| Aggregation | Accuracy | AUROC | Sensitivity | Specificity | F1 | ECE |
|---|---:|---:|---:|---:|---:|---:|
| mean | **0.8224** | **0.8783** | 0.7249 | **0.9017** | 0.7855 | 0.0267 |
| max | 0.7957 | 0.8750 | **0.7974** | 0.7943 | 0.7779 | 0.0213 |
| top2_mean | 0.8157 | 0.8772 | 0.7565 | 0.8638 | **0.7865** | **0.0193** |

### 6.2 ResNet50 Study-Level Results

| Aggregation | Accuracy | AUROC | Sensitivity | Specificity | F1 | ECE |
|---|---:|---:|---:|---:|---:|---:|
| mean | 0.8232 | **0.8819** | 0.6840 | **0.9365** | 0.7764 | **0.0156** |
| max | 0.8207 | 0.8806 | **0.7584** | 0.8714 | **0.7915** | 0.0266 |
| top2_mean | **0.8249** | 0.8809 | 0.7119 | 0.9168 | 0.7848 | 0.0208 |

Study-level aggregation improved performance relative to image-level evaluation.

For ResNet50:

```text
image-level accuracy: 0.8151
study-level top2_mean accuracy: 0.8249

image-level AUROC: 0.8741
study-level mean AUROC: 0.8819
```

The aggregation rule controlled the sensitivity-specificity tradeoff:

```text
mean aggregation:
sensitivity = 0.6840
specificity = 0.9365

max aggregation:
sensitivity = 0.7584
specificity = 0.8714

top2_mean aggregation:
sensitivity = 0.7119
specificity = 0.9168
```

This means that study-level aggregation is not just a technical detail. It directly changes the behavior of the classifier.

The main study-level finding was:

> Study-level aggregation better matched the structure of MURA and improved performance, but the aggregation rule acted like a clinical operating choice.

## 7. Study-Level Temperature Scaling

Temperature scaling was also learned directly at the study level. This was done because image-level calibration does not necessarily optimize calibration after aggregating multiple image probabilities into one study-level prediction.

The procedure was:

```text
image logits / temperature
→ sigmoid probabilities
→ aggregate probabilities into study prediction
→ optimize study-level negative log-likelihood
```

### 7.1 Study-Level Temperature Scaling Results

| Model | Aggregation | Study Temperature | Vanilla NLL | Scaled NLL | Vanilla ECE | Scaled ECE |
|---|---|---:|---:|---:|---:|---:|
| ResNet18 | mean | 1.00 | 0.4280 | 0.4280 | 0.0267 | 0.0267 |
| ResNet18 | max | 1.18 | 0.4558 | 0.4525 | 0.0213 | 0.0290 |
| ResNet18 | top2_mean | 1.04 | 0.4315 | 0.4314 | 0.0193 | 0.0273 |
| ResNet50 | mean | 1.14 | 0.4274 | 0.4250 | **0.0156** | 0.0191 |
| ResNet50 | max | 1.17 | 0.4329 | 0.4292 | 0.0266 | **0.0188** |
| ResNet50 | top2_mean | 1.12 | 0.4225 | **0.4207** | 0.0208 | 0.0200 |

Study-level temperature scaling improved negative log-likelihood across all ResNet50 aggregation methods:

```text
mean NLL:      0.4274 → 0.4250
max NLL:       0.4329 → 0.4292
top2_mean NLL: 0.4225 → 0.4207
```

ECE improvements were mixed. For ResNet50 max aggregation, ECE improved from 0.0266 to 0.0188. For ResNet50 mean aggregation, ECE worsened slightly from 0.0156 to 0.0191.

This result is useful because it shows that calibration depends on the prediction level and the aggregation rule. Optimizing one calibration objective does not guarantee improvement across every calibration metric.

The main study-level calibration finding was:

> Calibration should be evaluated at the same level where predictions are deployed. Image-level calibration and study-level calibration are related but not identical.

## 8. Test-Time Augmentation Uncertainty

Test-time augmentation was used as a lightweight uncertainty-estimation method.

Each validation image was evaluated under eight deterministic views:

```text
original
brightness down
brightness up
contrast down
contrast up
rotation -5 degrees
rotation +5 degrees
slight center crop
```

The mean probability across TTA views was used for classification. The variation across TTA predictions was used as an uncertainty signal.

Three uncertainty metrics were evaluated:

```text
probability standard deviation
entropy
margin uncertainty
```

Entropy measures how close the mean prediction is to uncertainty. A probability near 0.5 has high entropy, while a probability near 0 or 1 has low entropy.

Margin uncertainty similarly measures closeness to the decision boundary.

### 8.1 TTA Classification Results

| Model | Accuracy | AUROC | F1 | ECE |
|---|---:|---:|---:|---:|
| ResNet18 original | 0.8001 | 0.8688 | 0.7756 | 0.0202 |
| ResNet18 + TTA | **0.8017** | **0.8708** | **0.7835** | **0.0153** |
| ResNet50 original | 0.8151 | 0.8741 | 0.7833 | 0.0370 |
| ResNet50 + TTA | **0.8183** | **0.8745** | **0.7894** | **0.0191** |

TTA slightly improved classification performance and calibration for both models.

### 8.2 TTA Error-Detection Results

| Model | Std Error AUROC | Entropy Error AUROC | Margin Error AUROC |
|---|---:|---:|---:|
| ResNet18 | 0.6906 | **0.7375** | **0.7375** |
| ResNet50 | 0.6907 | **0.7167** | **0.7167** |

Incorrect predictions had higher uncertainty than correct predictions.

For ResNet50:

```text
correct entropy:    0.3630
incorrect entropy:  0.5076
gap:                0.1446
```

For ResNet18:

```text
correct entropy:    0.3967
incorrect entropy:  0.5608
gap:                0.1641
```

This means uncertainty was not random. The model was more uncertain on many of its mistakes.

The main TTA finding was:

> TTA uncertainty provided a useful error signal. It did not perfectly identify all mistakes, but it helped rank predictions by risk.

## 9. Robustness Under Synthetic Corruptions

The models were evaluated under synthetic corruptions to test whether calibration and confidence remained reliable under distribution shift.

Corruptions included:

```text
Gaussian noise
blur
brightness down
brightness up
contrast down
contrast up
```

### 9.1 Selected ResNet50 Corruption Results

| Condition | Severity | Accuracy | AUROC | F1 | ECE | Mean Confidence |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0 | 0.8151 | 0.8741 | 0.7833 | 0.0370 | 0.8425 |
| gaussian_noise | 4 | 0.5317 | 0.4456 | 0.0507 | 0.1798 | 0.7116 |
| blur | 4 | 0.6325 | 0.6997 | 0.6154 | 0.0423 | 0.6748 |
| brightness_up | 4 | 0.7792 | 0.8377 | 0.7328 | 0.0497 | 0.8288 |
| contrast_up | 4 | 0.7911 | 0.8458 | 0.7539 | 0.0563 | 0.8414 |

Severe Gaussian noise caused the largest reliability failure. AUROC fell below 0.5, F1 dropped to 0.0507, and ECE increased to 0.1798. Despite this, mean confidence remained 0.7116.

This is one of the strongest cautionary results in the project:

> A model can appear well-calibrated on clean validation data but become confidently wrong under distribution shift.

Blur caused a more expected degradation: performance dropped and confidence also decreased. Brightness and contrast shifts produced smaller performance drops but sometimes maintained high confidence.

The main robustness finding was:

> Reliability must be tested under shifted conditions, not only clean validation data.

## 10. Threshold Analysis

The default decision threshold of 0.5 was not always optimal.

For ResNet50, threshold tuning revealed that the lower sensitivity at threshold 0.5 was partly an operating-point issue.

| Probability Type | Operating Point | Threshold | Accuracy | Sensitivity | Specificity | F1 |
|---|---|---:|---:|---:|---:|---:|
| vanilla | default | 0.50 | 0.8151 | 0.6980 | 0.9226 | 0.7833 |
| vanilla | best F1 | 0.30 | 0.8167 | 0.8007 | 0.8314 | **0.8070** |
| temperature scaled | best F1 | 0.35 | **0.8183** | 0.7895 | 0.8446 | 0.8061 |

At threshold 0.5, ResNet50 was highly specific but less sensitive. Lowering the threshold improved sensitivity and F1, at the cost of more false positives.

For a medical screening context, a lower threshold may be preferred if missing abnormalities is considered more costly than flagging normal studies.

The main threshold finding was:

> Classification behavior depends strongly on the selected threshold, so model evaluation should report operating-point tradeoffs rather than only default-threshold metrics.

## 11. Failure-Case Analysis

Failure-case visualization was used to inspect examples where the calibrated ResNet50 model was wrong.

Three types of cases were especially useful:

```text
high-confidence false negatives
high-confidence false positives
uncertain wrong predictions
```

High-confidence false negatives were the most concerning. These were abnormal X-rays predicted as normal with high confidence. This type of error is clinically important because it represents a missed abnormality.

High-confidence false positives often showed unusual positioning, acquisition artifacts, hardware, or high-contrast appearances. These cases suggested that the model may sometimes rely on non-pathological visual cues.

Uncertain wrong predictions were less concerning because the model was close to the decision boundary. These are examples where uncertainty behaved more appropriately.

The main failure-case finding was:

> Calibration and uncertainty improve aggregate reliability, but they do not remove all high-confidence errors. Individual failure inspection remains necessary.

## 12. Overall Findings

The project produced several main findings.

### 12.1 ResNet50 was the strongest classifier, but not automatically the most reliable

ResNet50 had the best image-level accuracy, AUROC, precision, specificity, and F1. However, it had worse calibration than ResNet18 before temperature scaling.

This shows that model selection should consider both performance and reliability.

### 12.2 Temperature scaling improved clean calibration

Temperature scaling reduced ResNet50 image-level ECE from 0.0370 to 0.0094 and improved NLL. This made ResNet50 the strongest overall clean model after calibration.

### 12.3 Study-level aggregation improved evaluation quality

MURA labels are study-level labels. Aggregating image probabilities to the study level improved accuracy and AUROC and better matched the dataset structure.

### 12.4 Aggregation rules changed clinical behavior

Mean aggregation was more specific and better calibrated. Max aggregation improved sensitivity and F1. Top-2 mean provided a balanced alternative.

This means aggregation is part of the model’s operating design, not just a reporting detail.

### 12.5 Study-level calibration behaved differently from image-level calibration

Study-level temperature scaling improved ResNet50 NLL across aggregation methods, but ECE improvements depended on the aggregation rule.

This shows that calibration should be evaluated at the final prediction level.

### 12.6 TTA uncertainty helped flag risky predictions

TTA uncertainty was higher on incorrect predictions than correct predictions. Entropy-based error detection reached AUROC 0.7375 for ResNet18 and 0.7167 for ResNet50.

This suggests that uncertainty can help identify predictions that may need review.

### 12.7 Corruption robustness exposed hidden failure modes

Severe Gaussian noise caused major performance collapse while confidence remained high. This failure was not visible from clean validation metrics alone.

### 12.8 High-confidence false negatives remained

Even after calibration and uncertainty analysis, some abnormal X-rays were predicted as normal with high confidence.

This is the most important safety limitation.

## 13. Limitations

This project has several important limitations.

First, it is not a clinical diagnostic system. The models were evaluated in a research setting and were not validated for clinical use.

Second, most experiments used the MURA validation set rather than an external dataset. External validation would be needed to evaluate generalization across hospitals, scanners, acquisition protocols, and patient populations.

Third, although study-level aggregation was added, the models themselves were trained using image-level inputs. A more complete study-level model could explicitly process all images in a study together.

Fourth, the corruption tests used synthetic corruptions. These are useful stress tests but do not fully represent real clinical distribution shifts.

Fifth, temperature scaling was learned and evaluated on the available validation distribution. A more rigorous setup would use separate calibration and evaluation splits.

Sixth, the project used standard CNN backbones. More specialized medical imaging architectures or modern vision transformers could be explored in future work.

Seventh, failure cases were visually inspected but not reviewed by clinicians.

Finally, uncertainty signals were useful but imperfect. Some wrong predictions still had high confidence or low uncertainty.

## 14. Future Work

Several extensions could strengthen this project.

### 14.1 External validation

Testing on another musculoskeletal X-ray dataset would better evaluate generalization beyond MURA.

### 14.2 Multi-seed training

Running multiple seeds would allow stronger claims about result stability.

### 14.3 Larger or alternative backbones

DenseNet, EfficientNet, ConvNeXt, or vision transformers could be compared against the ResNet baselines.

### 14.4 Higher-resolution training

X-ray abnormalities can be subtle. Higher input resolution may improve performance but would require more compute.

### 14.5 Study-level model architecture

Instead of aggregating image probabilities after inference, a model could directly consume multiple images from the same study.

### 14.6 Body-part-specific calibration

Different body parts may have different calibration behavior. Wrist, shoulder, and finger studies may not have identical uncertainty patterns.

### 14.7 Deep ensembles or MC dropout

Additional uncertainty methods could be compared against TTA uncertainty.

### 14.8 Clinician review

Radiologist or clinician review of high-confidence failures would make the failure-case analysis more meaningful.

## 15. Conclusion

This project showed that musculoskeletal X-ray abnormality classifiers should be evaluated beyond accuracy alone.

ResNet50 achieved the strongest classification performance, but reliability analysis revealed that it was initially more overconfident than ResNet18. Temperature scaling improved clean calibration, study-level aggregation better matched the MURA dataset structure, and TTA uncertainty provided a useful signal for identifying likely errors.

At the same time, corruption testing and failure-case analysis showed that reliability remains fragile. Severe Gaussian noise caused confident failure, and high-confidence false negatives remained even after calibration.

The main takeaway is:

> Reliable medical image classification requires evaluating not only whether a model is correct, but also whether its confidence, uncertainty, and robustness behave sensibly when the model is wrong or when the input distribution changes.