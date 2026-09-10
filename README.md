# CNN-on-Balding-Data-Set

## Description of Data Set
The models are trained and evaluated on two public Kaggle datasets: a CelebA-derived set (ashishjangra27/bald-classification-200k-images-celeba), introduced by Liu et al. (2015), of roughly 200,000 aligned 178×218 celebrity faces pre-labelled bald or not-bald, and the HairLoss set (sithukaungset/hairlossdataset) of 1,114 images with the same binary labels at varied resolutions. The CelebA bald attribute has a positive rate of roughly 2 percent and inherits crowdsourced label noise, which is acknowledged as a limitation. 

<div align="center">
  <img width="461" height="127" alt="image" src="https://github.com/user-attachments/assets/894b370c-9a94-4e98-91d1-5aa25725d5ed" />
  <p><em>Table 1: Dataset Overview (exact counts reported before filtering).</em></p>
</div>


Rather than the full CelebA set, a subsample was used: 5,000 not-bald images from each of the three existing CelebA splits (15,000 total) with the bald class kept in full (4,547 images), which raises the positive rate to roughly 25 percent and prevents CelebA from dominating the much smaller HairLoss set. The 560 HairLoss bald images were manually reviewed and mislabelled, multi-person, composite, and unusable samples removed.
The collected images varied widely in size, which would introduce noise if left unnormalised, and the two sources differ slightly in style and variety. Figure 1 depicts the distribution of image sizes across the gathered data from kaggle. The images from CelebA had consistent dimensions (178x218), while the HairLoss dataset had very dispersed image sizes.
<div align="center">
  <img width="380" height="317" alt="image" src="https://github.com/user-attachments/assets/83a601da-5ff0-4f1c-8937-fa40a8bf93a1" />
  <p><em>Figure 1: Image height (pixels) against width (pixels). Blue and orange points represent balding and not balding images, respectively. The red line represents square images.</em></p>
</div>

The images from CelebA are also tightly cropped, front-facing celebrity portraits with consistent dimensions and a relatively uniform style. In contrast to the HairLoss images which differ in framing, pose and style (pictures taken from above, side, etc.)

## Description of EDA & Integrity Checks

A four-stage filtering pipeline is applied before any splitting or modelling, with removal counts in Table 2. 
<div align="center">
  <img width="550" height="140" alt="image" src="https://github.com/user-attachments/assets/6f5d96c3-3633-4e00-844b-d9f2f7975441" />
  <p><em>Table 2: Integrity Checks</em></p>
</div>

Stage one verifies file integrity by attempting to open each image with PIL and then manually reviewing the HairLoss bald data set, resulting in 155 removed images.

Stage two detects duplicates by perceptual hashing (phash) (see Table 2). Any images with a hamming distance less than 3 were declared duplicates. Average hashing was also calculated for both balding and notbalding data sets, to detect more images, potentially missed by phash. However, average hashing was only used for balding images, as it declared different images “similar” in the notbalding data set. Pairs across the balding and notbalding data sets were also checked and removed, to reduce data leakage.

Stage three evaluates per-channel RGB and overall brightness. EDA confirmed that all images were valid RGB at and no RGB anomalies were found. Brightness differed systematically between sources (CelebA mean 112.14, HairLoss 132.41) but only slightly between classes (bald 115.40, not-bald 112.59) (see [Figure 2](#contrast-brightness-source)). The between-source difference of roughly 20 intensity units is an order of magnitude larger than the between-class difference of roughly 3, so a model could exploit source-correlated brightness rather than class-relevant features, which motivates the source-balanced split. Inspection of the brightest images surfaced three animated drawings in the bald class, which were removed, leaving 4,612 bald images. 

Stage four evaluates pixel-intensity contrast, which is close across both sources (66.33 against 67.78) and classes (64.81 against 66.78). No images were removed on contrast, but the statistics are reported to characterise the data. 
<div align="center" id = "contrast-brightness-source">
  <img width="592" height="250" alt="image" src="https://github.com/user-attachments/assets/de30b62c-e3a2-4ee4-9882-884abe99eab1" />
  <p><em>Figure 2: Brightness and contrast differences split on source of image.</em></p>
</div>

## Train, Validation, and Test Split

The data is partitioned 60/20/20 with a fixed seed of 42. Stratification was done to preserve both class balance, and source balance. Yielding an approximate 23% balding rate, and 96% CelebA rate, across data sets. The index assignments are written and consumed identically by every model, so cross-model comparisons remain valid. 

## Data Normalisation and Augmentation
All inputs are resized to 224×224 and rescaled to [0, 1], with ImageNet channel-wise normalisation additionally applied for the pretrained ResNet50. Augmentation is applied only to the training subset, and balding images, keeping test and validation images real and increasing the minority class. Two transformations are used: random horizontal flip (justified by the bilateral symmetry of the face) and additive Gaussian noise at σ=0.05 (simulating sensor variation). Each is applied to every bald training image, tripling the bald subset and shifting the effective training distribution from 23% to 47% bald. Color jitter was implemented and tested but disabled in the final pipeline because horizontal flip and Gaussian noise alone produced the most stable training dynamics for the CNN. Augmentation magnitudes are otherwise kept small so as to not remove important features from the data set. [Figure 3](#augmentation) shows an example of the augmentations applied to training images.
<div align="center" id="augmentations">
  <img width="552" height="149" alt="image" src="https://github.com/user-attachments/assets/2d3665fd-51cc-4a9b-980a-0c672eadf61c" />
  <p><em>Figure 3: Data augmentation applied to exemplary image</em></p>
</div>

## CNN
The initial CNN structure consisted of 3 Conv2D layers, followed by batch normalization, ReLU activation, and max pooling. The three convolutional layers contained 16, 32, and 64 filters, respectively, each using a kernel size of 3 × 3. This kernel size was selected to balance computational efficiency with the ability to extract local spatial features. Following the convolutional blocks, the feature maps were flattened and passed to a fully connected layer with 256 neurons, followed by a single-neuron output layer to produce a probabilistic binary prediction.

Several iterative experiments were conducted to improve model performance and reduce overfitting. These included the introduction of dropout, reduction of the learning rate, adjustment of batch size to 64, and modification of the number of training epochs to account for the reduced number of weight updates per epoch. In addition, callback functions were implemented, including model checkpointing, early stopping, and learning rate reduction on plateau. Different combinations of image augmentation techniques were also evaluated, along with modifications to the network depth and the use of a weighted binary cross-entropy loss function to penalize misclassification of positive (bald) images more heavily.

The final model architecture (see [figure 4](#finalcnn)) consisted of four convolutional blocks with 16, 32, 64, and 128 filters, respectively, all using 3 × 3 kernels. Each block was followed by batch normalization, ReLU activation, and MaxPooling2D. The convolutional layers were followed by a fully connected layer with 256 neurons and a dropout layer with a rate of 0.4. The output layer contained a single neuron with sigmoid activation for binary classification.

The model was trained using the Adam optimizer with a learning rate of 1 × 10⁻⁴. Weighted binary cross-entropy was used as the loss function, with the positive class weight calculated as the ratio of negative to positive samples plus 1, resulting in a value of 2.11. Training employed three callback functions: early stopping with a patience of 15 epochs, ReduceLROnPlateau with a patience of 5 epochs, and model checkpointing to retain the best-performing model.

<div align="center" id="finalcnn">
  <img width="697" height="321" alt="image" src="https://github.com/user-attachments/assets/6890ff13-b426-4e91-8385-2e59e439557f" />
  <p><em>Figure 4: Final CNN Design Summarised </em></p>
</div>

## Assignment Context

This repository contains the custom convolutional neural network (CNN) component of a machine-learning assignment on binary baldness classification from ordinary facial photographs. The practical objective is a first-stage screening signal: classify an image as **bald** or **not bald** before any later human assessment. The CNN is trained from scratch and is the focus of this repository. The wider assignment used the same fixed data partitions to contextualise the CNN against other approaches, but those other implementations and their results are not part of this CNN description.

## CNN Evaluation

The held-out test set contains 3,995 images and approximately 23% bald examples. Because a majority-class prediction already gives high accuracy on an imbalanced split, macro-F1 and bald-class precision and recall are more informative than accuracy alone. The CNN used a validation-selected decision threshold of 0.583.

| Metric | Custom CNN |
| --- | ---: |
| Macro-F1 | 0.91 |
| Bald precision | 0.80 |
| Bald recall | 0.94 |
| Bald F1 | 0.86 |
| Validation ROC-AUC | 0.978 |
| Training time | approximately 16 minutes |
| Inference time | approximately 0.8 ms per image |

On the test set, the CNN correctly identified 868 of 922 bald images and missed 54 (false-negative rate 5.9%). It incorrectly flagged 219 of 3,073 not-bald images as bald (false-positive rate 7.1%). The high bald recall is useful for a screening stage, while the lower bald precision means that positive predictions still require follow-up review.

## Interpretation and Error Analysis

The training accuracy increased from 0.73 to 0.89 over 20 epochs while training loss decreased from 0.77 to 0.30. Validation behaviour was more variable early in training, but validation loss reached 0.2578 at the final epoch. Batch normalization, dropout, weighted loss, and minority-class augmentation helped limit the gap between training and validation performance.

Grad-CAM inspection of the final convolutional layer indicates that the CNN learned a meaningful visual cue: scalp visibility around the crown. Activations for confident bald predictions were concentrated near the top of the head rather than on the background or image framing. This also explains an important failure mode: when a hat or thick hair hides the crown, the network lacks its main positive cue and may default to not-bald. Some apparent false positives also reflect label noise, unusual viewpoints, multiple people, or differences between the CelebA and HairLoss image sources.

For context, the same-input comparison in the assignment showed that the custom CNN improved on the non-neural baselines but remained below a fine-tuned pretrained ResNet50. This is consistent with the CNN learning useful task-specific features from scratch while the pretrained model starts with a richer visual representation; it does not change the custom CNN's standalone result above.

## Limitations and Future Work

The working data is a curated subsample with an artificially increased bald prevalence of about 23%, rather than the roughly 2% prevalence of the original CelebA bald attribute. Precision may therefore be substantially lower in deployment, and performance at the natural prevalence was not measured. CelebA annotations are crowdsourced and can contain label noise, while the two sources differ in brightness, framing, pose, and image style despite source-aware stratification.

The final CNN evaluation represents one selected imbalance strategy: threefold augmentation of bald training images together with weighted binary cross-entropy. A systematic comparison with undersampling, alternative class weights, focal loss, and other sampling strategies would clarify whether the remaining precision gap is intrinsic to the architecture or caused by the training distribution. Additional occlusion-focused augmentation and a graded hair-loss target could also address the crown-occlusion failure mode. Evaluation on a new, naturally prevalent dataset is needed before using the model for real screening.

## References

- Benhabiles, H., Hammoudi, K., Yang, Z., Windal, F., Melkemi, M., Dornaika, F., & Arganda-Carreras, I. (2019). *Deep learning based detection of hair loss levels from facial images*. 2019 Ninth International Conference on Image Processing Theory, Tools and Applications, 1–6.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 770–778. https://doi.org/10.1109/CVPR.2016.90
- Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). *Deep learning face attributes in the wild*. Proceedings of the IEEE International Conference on Computer Vision, 3730–3738.
- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization*. Proceedings of the IEEE International Conference on Computer Vision, 618–626. https://doi.org/10.1109/ICCV.2017.74
- The primary data sources are the [CelebA-derived bald classification dataset](https://www.kaggle.com/datasets/ashishjangra27/bald-classification-200k-images-celeba) and the [HairLoss dataset](https://www.kaggle.com/datasets/sithukaungset/hairlossdataset).
