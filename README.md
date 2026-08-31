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
  <img width="550" height="173" alt="image" src="https://github.com/user-attachments/assets/64814d75-e252-4ffc-b3d2-a7684adbd553" />
  <p><em>Table 2: Integrity Checks</em></p>
</div>
Stage one verifies file integrity by attempting to open each image with PIL and then manually reviewing the HairLoss bald data set, resulting in 155 removed images.

Stage two detects duplicates by perceptual hashing (phash) (see Table 2). Any images with a hamming distance less than 3 were declared duplicates. Average hashing was also calculated for both balding and notbalding data sets, to detect more images, potentially missed by phash. However, average hashing was only used for balding images, as it declared different images “similar” in the notbalding data set. Pairs across the balding and notbalding data sets were also checked and removed, to reduce data leakage.

Stage three evaluates per-channel RGB and overall brightness. EDA confirmed that all images were valid RGB at and no RGB anomalies were found. Brightness differed systematically between sources (CelebA mean 112.14, HairLoss 132.41) but only slightly between classes (bald 115.40, not-bald 112.59) (see [Figure 2](#contrast-brightness-source))(another href way <a href="#contrast-brightness-source"> figure 2</a>. The between-source difference of roughly 20 intensity units is an order of magnitude larger than the between-class difference of roughly 3, so a model could exploit source-correlated brightness rather than class-relevant features, which motivates the source-balanced split. Inspection of the brightest images surfaced three animated drawings in the bald class, which were removed, leaving 4,612 bald images (Appendix I). 

Stage four evaluates pixel-intensity contrast, which is close across both sources (66.33 against 67.78) and classes (64.81 against 66.78). No images were removed on contrast, but the statistics are reported to characterise the data. 
<div align="center" id = "contrast-brightness-source">
  <img width="592" height="250" alt="image" src="https://github.com/user-attachments/assets/de30b62c-e3a2-4ee4-9882-884abe99eab1" />
  <p><em>Figure 2: Brightness and contrast differences split on source of image.</em></p>
</div>
