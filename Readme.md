# Animal Image Classifier using PyTorch

A deep learning image classification project that uses **Transfer Learning with ResNet18** to classify images into three categories:

* Cats
* Dogs
* Panda

The trained model is deployed as an interactive **Streamlit web application**, where users can upload an image and receive a predicted class along with its confidence score.


---

##  Project Overview

Training an image classification model from scratch requires a large amount of data and computational resources.

Instead of training an entire convolutional neural network from scratch, this project uses **Transfer Learning**.

A pretrained **ResNet18** model trained on ImageNet is used as the feature extractor. Its original 1000-class classification layer is replaced with a custom classifier designed for the three target classes.

### Model Architecture

```text
Input Image
     ↓
Resize to 224 × 224
     ↓
ImageNet Normalization
     ↓
Pretrained ResNet18
     ↓
512 Features
     ↓
Linear Layer (512 → 256)
     ↓
ReLU
     ↓
Dropout (0.5)
     ↓
Linear Layer (256 → 3)
     ↓
Cat / Dog / Panda
```

The pretrained convolutional layers are frozen during training, w

## Live Demo

**Streamlit App:**
https://aiclassifierofdogcatpandausingpytorchgit-l7tavxe6bexxrwapmauvv.streamlit.app/