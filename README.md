# histosnet
This is the project for my ms thesis. The goal is to find new biomarkers for local recurrence in cases of triple-negative breast cancer. A possible biomarker is the spatial distribution of lymphocytes. To investigate this I'm using a U-Net trained on public datasets to segment and classify cells on whole-slide images of breast cancer from the SweBCG91RT study. 

## Data
Multiple datasets have used in this project. I've used public data to train the U-Net. The trained model has then been used for cell segmentation on images from the TNBC cases.

### [MoNuSeg](https://monuseg.grand-challenge.org/Data/)
This dataset was used in the 2028 MoNuSeg Challenge.
* 30 images of cancer tissue from various organs
* 1000 x 1000 pixels
* 40x magnification
* 22,000 cells annotated.

### [Bns](http://members.cbio.mines-paristech.fr/~pnaylor/BNS.zip)
This dataset was used by [(N. Peter 2017)](https://ieeexplore.ieee.org/document/7950669). Way fewer cells in this dataset, on the other hand it's much closer to the target domain.
* 33 images of triple-negeative breast cancer tissue. The annotations can be found [here](https://wiki.cancerimagingarchive.net/display/DOI/Dataset+of+Segmented+Nuclei+in+Hematoxylin+and+Eosin+Stained+Histopathology+Images) and the WSIs can be found at [The Cancer Genome Atlas](https://cancergenome.nih.gov/)
* 512 x 512 pixels
* 40x magnification
* 2758 cells annotated.

### [Quip]()
This dataset was created by [(L. Hou et al. 2020)](https://www.nature.com/articles/s41597-020-0528-1). It's very large.
* 5 billion annotated cells for various cancer types, breast included.
* 40x magnification.
* consists of WSIs but the annotations are made in 4000 x 4000 patches.
