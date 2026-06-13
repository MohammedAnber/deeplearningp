# Research Log — CIFAR-10 Grad-CAM Project

## Project Overview

This project implemented Grad-CAM from scratch, trained three CIFAR-10 classifiers, validated the implementation against a reference library, and analyzed what the heatmaps reveal about each model’s learned representations. The three models were BaselineCNN, ResNet18-scratch, and ResNet18-pretrained. The final write-up emphasizes both predictive performance and explanation quality.

## Motivation and Scope

The goal was not only to obtain strong classification accuracy, but also to test whether the explanation method is technically correct, stable under sanity checks, and meaningful under model comparison. This was evaluated through Grad-CAM parity tests, Adebayo-style randomization checks, and formal interpretability metrics. Recent XAI literature increasingly frames explanation quality around faithfulness, robustness, consistency, and localization quality, which matches the structure of this project’s evaluation [web:100][web:102][web:108][web:111].

## Recent Literature

Recent work on XAI evaluation has moved beyond purely qualitative heatmap inspection and now emphasizes quantitative metrics, explanation robustness, and consistency under perturbations [web:100][web:102][web:108][web:111]. Studies published in 2023–2026 specifically discuss SSIM-based comparison, perturbation-driven evaluation, localization reliability, and the need for model-aware explanation assessment rather than visual intuition alone [web:99][web:100][web:102][web:108]. This project follows that direction by using both similarity-based and concentration-based metrics, plus sanity checks that test whether explanations depend on learned model parameters [web:99][web:102][web:104].

## Dataset

We used CIFAR-10, which contains 10 balanced classes with 50,000 training images and 10,000 test images. The dataset was verified to be perfectly class-balanced, and the preprocessing pipeline was designed to avoid leakage by computing normalization statistics from the training set only. Because CIFAR-10 images are low-resolution, explanation figures can become visually dense, so final report figures should be enlarged and spaced more carefully [file:91].

## Preprocessing

Training augmentation used random cropping with padding and horizontal flipping. No vertical flip was used because CIFAR-10 contains natural images, not upside-down scenes. Inputs were normalized using training-set mean and standard deviation. This preprocessing kept the pipeline standard while preserving object semantics.

## Architecture

### BaselineCNN
A custom three-block CNN with convolution, batch normalization, ReLU, max pooling, and a small classifier head. It has 620,810 trainable parameters. This model serves as the lightweight baseline.

### ResNet18-scratch
A CIFAR-10-adapted ResNet18 trained from random initialization. The first convolution was changed to 3x3 stride 1, and the initial max-pooling layer was removed. It has 11,173,962 trainable parameters.

### ResNet18-pretrained
The same CIFAR-10-adapted ResNet18 initialized from ImageNet pretrained weights. It also has 11,173,962 trainable parameters. Transfer learning was expected to improve convergence speed and final accuracy.

## Training Setup

All experiments used SGD with momentum 0.9, weight decay 5e-4, cosine annealing, batch size 128, and 100 epochs. A fixed seed of 42 was used for reproducibility. This setup was chosen for stable convergence and fair comparison across models.

## BaselineCNN Training Results

| Epoch | Train Loss | Val Loss | Val Acc | Gen Gap |
|-------|------------|----------|---------|---------|
| 1     | 1.5994     | 1.1797   | 56.6%   | -0.4197 |
| 50    | 0.4243     | 0.4511   | 84.3%   | +0.0268 |
| 100   | 0.2907     | 0.3824   | 87.0%   | +0.0917 |

Best validation accuracy: 87.10%.

The baseline learned steadily but remained weaker than the deeper models. Its explanations were also less semantically consistent, which is expected from a shallower CNN.

## ResNet18-scratch Training Results

| Epoch | Train Loss | Val Loss | Val Acc | Gen Gap |
|-------|------------|----------|---------|---------|
| 1     | 1.5556     | 1.1896   | 57.5%   | -0.3660 |
| 50    | 0.0289     | 0.3484   | 91.2%   | +0.3195 |
| 100   | 0.0021     | 0.2836   | 92.9%   | +0.2815 |

Best validation accuracy: 92.93%.

This model achieved stronger performance than the baseline, but with a noticeably larger generalization gap. It learned the training set very aggressively, which suggests stronger representational capacity but also more overfitting.

## ResNet18-pretrained Training Results

| Epoch | Train Loss | Val Loss | Val Acc | Gen Gap |
|-------|------------|----------|---------|---------|
| 1     | 0.7730     | 0.4699   | 84.6%   | -0.3031 |
| 50    | 0.0072     | 0.2048   | 94.7%   | +0.1976 |
| 100   | 0.0009     | 0.1540   | 96.2%   | +0.1530 |

Best validation accuracy: 96.16%.

Transfer learning gave the best final performance and the fastest convergence. Its heatmaps were also the most semantically meaningful and most stable across classes.

## Evaluation Results

| Model | Test Accuracy | Avg Loss |
|-------|---------------|----------|
| BaselineCNN | 87.10% | 0.3817 |
| ResNet18-scratch | 92.93% | 0.2829 |
| ResNet18-pretrained | 96.16% | 0.1550 |

The pretrained model performed best on the test set, followed by the scratch ResNet, then BaselineCNN. This ranking is consistent with the training curves and the later interpretability results.

## Grad-CAM Implementation

Grad-CAM was implemented manually using forward and full-backward hooks on the target layer. The backward pass used raw logits rather than softmax to avoid class competition, and a ReLU was applied after the weighted sum of feature maps so that only positive evidence contributed to the final heatmap. This design matches standard Grad-CAM practice and supports faithful localization [web:104].

## Quantitative XAI Metrics

### Definitions

Let \(H_i\) and \(H_j\) be normalized heatmaps.

\[
\mathrm{LCS} = \frac{1}{N} \sum_{(i,j)} \mathrm{SSIM}(H_i, H_j)
\]

where \(N\) is the number of heatmap pairs for the same class. LCS measures how consistently a model highlights the same region across examples of the same class [web:100][web:102][web:108].

\[
\mathrm{EEC}_p(H) = \frac{\sum_{k \in \text{Top-}p\%} H_k}{\sum_{k} H_k}
\]

EEC measures how concentrated the explanation energy is in the top \(p\%\) most relevant pixels [web:106][web:108].

\[
\mathrm{ICD} = 1 - \frac{1}{M} \sum_{(a,b)} \mathrm{SSIM}(H_a, H_b)
\]

where the sum is over inter-class heatmap pairs. ICD measures how distinct explanations are across classes [web:99][web:100].

\[
\mathrm{RS}_k = 1 - \mathrm{SSIM}(H_0, H_k)
\]

where \(H_0\) is the original heatmap and \(H_k\) is the heatmap after progressive randomization. RS checks whether explanations change when learned weights are destroyed [web:104][web:111].

### Final Metric Summary

| Model | LCS Mean | EEC Mean | ICD | RS Monotonic? |
|-------|----------|----------|-----|---------------|
| BaselineCNN | 0.1476 | 0.4836 | 0.8452 | Yes |
| ResNet18-scratch | 0.7293 | 0.4261 | 0.2852 | No |
| ResNet18-pretrained | 0.7938 | 0.4197 | 0.1827 | No |

These results show that the deeper models produced more consistent heatmaps within class, while the baseline produced more concentrated but less semantically stable explanations. The pretrained model achieved the strongest within-class consistency and the lowest inter-class confusion in explanation space.

## Per-Class LCS

| Class | BaselineCNN | ResNet18-scratch | ResNet18-pretrained |
|-------|-------------|------------------|---------------------|
| Airplane | 0.083 | 0.813 | 0.673 |
| Automobile | 0.281 | 0.720 | 0.785 |
| Bird | 0.112 | 0.635 | 0.765 |
| Cat | 0.241 | 0.755 | 0.843 |
| Deer | 0.121 | 0.580 | 0.767 |
| Dog | 0.134 | 0.721 | 0.913 |
| Frog | 0.081 | 0.770 | 0.755 |
| Horse | 0.144 | 0.780 | 0.756 |
| Ship | 0.040 | 0.717 | 0.815 |
| Truck | 0.239 | 0.801 | 0.867 |
| Mean | 0.148 | 0.729 | 0.794 |

The pretrained model is strongest for most classes, especially dog, cat, and truck. The baseline remains much less stable across classes, which is consistent with its shallower feature hierarchy.

## Library Parity Validation

| Model | Spearman r | Pass? |
|-------|------------|-------|
| BaselineCNN | 1.0000 | True |
| ResNet18-scratch | 1.0000 | True |
| ResNet18-pretrained | 1.0000 | True |

The manual Grad-CAM implementation matched the reference library exactly on the evaluated sample, which strongly supports correctness. This is an important validation step because the strong qualitative results would otherwise be hard to trust [web:104].

## Sanity Checks

Adebayo-style model randomization and data randomization tests were used to verify that the explanations depend on learned weights rather than dataset priors alone. Under shuffled labels, models stayed near 10% accuracy and cross-entropy remained near 2.30, which is consistent with random guessing. This supports the claim that the heatmaps are tied to learned representations rather than accidental image statistics [web:104][web:111].

## Interpretation

The pretrained model produced the best trade-off between classification accuracy and explanation quality. The scratch ResNet was also strong, but it showed a larger generalization gap. The baseline was simpler and more stable in some metrics, but its explanations were less semantically aligned with the learned classes. Overall, classification accuracy and explanation quality were positively related, but not identical.

## Timeline

- **Week 1:** Repository setup, dataset inspection, and preprocessing pipeline.
- **Week 2:** Hayam A. Rezk worked on architecture design, training infrastructure, and the first training pipeline runs.
- **Week 3:** Mohammed A. Anber worked on Grad-CAM from scratch and the first sanity checks.
- **Week 4:** Joint work on model training, evaluation, and Grad-CAM visualization refinement.
- **Week 5:** Hayam A. Rezk worked on all 100-epoch GPU runs, training curves, and dropout ablation.
- **Week 6:** Mohammed A. Anber worked on Grad-CAM++ comparison and the LCS/EEC/ICD/RS metrics and statistics. Joint work on experimental analysis and report writing.

## Visualization Notes

Some CIFAR-10 images are low-resolution, so Grad-CAM layouts can become visually dense. Final figures should use larger figure sizes, fewer subplots per page, and more spacing between panels to improve readability. This is especially important for side-by-side model comparisons and per-class heatmap grids [file:91].

## Conclusion

The project successfully combined accurate classification, faithful explanation generation, and rigorous validation. To fully satisfy the final-report rubric, the main remaining improvements are stronger recent literature coverage, explicit metric definitions, a more structured timeline, and clearer figure presentation.