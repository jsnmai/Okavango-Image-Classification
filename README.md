# Files

**1. partitioning.ipynb:** Loads NDJSON label file to a pandas dataframe, performs stratified sampling to create train/validation/test splits ensuring the proportion of each class in each split is the same as the in the original dataset, finally plots class distribution for each split for visualization and comparison.

**2. data_utils.py:** A DataPartition class and definitions for additional MixUp, CutMix, and RandomHorizontalRoll Image Transformations that must sit separately as a script and imported to the following Jupyter Notebooks for GPU training to work.

**3. training.ipynb:** Code for finetuning a pretrained model. Current version comes with and is tested for SwinV2 Transformer model. See "Resources" for choosing other models.

**4. classifier.ipynb:** Code for deploying and using a model trained from training.ipynb.


### Miscellaneous Files
- **labels.ndjson:** Labels exported from LabelBox used for training.
- **train_partition.csv:** Subset of labels for training.
- **val_partition.csv:** Subset of labels for validation stage.
- **test_partition.csv:** Subset of labels for test stage.
- **best_classifier.json:** Config file required to utilize the model weights.
- **best_model.pt:** Our best model from finetuning a pretrained SwinV2. [Download here.](https://drive.google.com/file/d/1JkRNOciNHzjb7xiSiIfo4t-cPYABm7lT/view?usp=drive_link)

# Resources

- [Implementation guide for finetuning a pretrained model](https://www.kaggle.com/code/gohweizheng/swin-transformer-beginner-friendly-notebook#1.-Introduction) (specifically Swin Transformer)

- [Pretrained Models to choose from timm](https://huggingface.co/models?library=timm)
  - [Benchmarks for choosing a pretrained Swin Transformer model](https://github.com/microsoft/Swin-Transformer/blob/main/MODELHUB.md#imagenet-22k-pretrained-swin-moe-models)
    (*benchmarks are for regular image classification on ImageNet dataset)
  - [Benchmarks for choosing/comparing other pretrained models](https://paperswithcode.com/sota/image-classification-on-imagenet)
    (*benchmarks are for regular image classification on ImageNet dataset)

- Optimal Prediction Thresholding Strategies:
  - https://www.evidentlyai.com/classification-metrics/classification-threshold
  - https://www.mathworks.com/help/deeplearning/ug/multilabel-image-classification-using-deep-learning.html
  - https://www.mdpi.com/2076-3417/13/13/7591

 # Getting Started
 ### Recommend using a virtual environment ([conda](https://www.anaconda.com/download) is common for ML) to contain all work/imported libraries
 Create conda env with:

    conda create -n YourNameHere python=3.11

To activate conda env:

    conda activate YourNameHere

   
