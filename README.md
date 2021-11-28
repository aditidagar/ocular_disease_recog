# Ocular Disease Inteligent Recognition

## Introduction: the problem

## The Dataset

## Previous efforts referenced

- [*A Benchmark of Ocular Disease Intelligent Recognition: One Shot for Multi-Disease Detection [1]* ](https://doi.org/10.1007/978-3-030-71058-3_11): This is the original paper associated with the ODIR-5k dataset, written by the authors of the dataset.
  
  -  It establishes baseline benchmarks for the dataset by using eight deep convolutional neural networks (CNN): Vgg-16, ResNet-18, ResNet-50, ResNeXt-50, SE-ResNet-50, SE-ResNeXt-50, Inception-v4, Densenet and CaffeNet. These CNN's were referenced in inspiring our choice of CNN.
  
  - In the authors model, left and right fundus images are fed into seperate models, and a feature fusion method (element-wise sum, element-wise multiplication or concatenation) is used to combine the output of both models. The authors emphasize that an occular disease recognition model should evaluate both the left and right fundus images to provide correct diagnosis. Rather than implementing a feature fusion technique, we decided to optionally fuse left and right fundus images as part of our pre-processing pipeline.  

- [*Algorithms For Massive Data [2]*](https://github.com/GrzegorzMeller/AlgorithmsForMassiveData): Our model was initially forked from the Grzegorz Meller's model. His model was chosen as a starting point due to it's relatively simple implementation (which aided us in understanding the entire model).
  
  -  It is spread across a series of Jupyter-notebook's hosted on Google Colab, which are easy to quickly test and run.
  
  -  We credit Grzegorz Meller for his original code which is used in our pre-processing and augmentation pipeline (though with improvements) as well as the initial setup for our various models. 

- [*Ocular Disease Intelligent Recognition Through Deep Learning Architectures [3]*](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning): Jordi Corbilla's model was referenced in helping us pick ML CNN's and used as a guideline to implement them for our dataset.  

## Machine Learning Environment

## Our model

Before feeding the ODIR-5k dataset to the ML model, we run the data through a pre-processing and augmentation pipeline. First, we pre-process the training data from the dataset through image enhancement. This is followed by the creating a validation set by randomly sampling 30% of the training data. Afterwards, the training data is augmented to even the distribution of images per-disease. Finally, the data is given as an input to one of the 4 models implemented.  

The figure below depicts the pre-processing and augmentation pipeline, which will be subsequently explained.

### Pre-processing

The pre-processing pipeline consists of the following steps:

1. **Deletion of unnecessary data:** the dataset contains images which are marked as low quality, or containing artifacts such as camera lense flare. These are removed from the training dataset to prevent them from giving incorrect bias to the ML model. 

2. **Appending disease labels to image filenames:**  In order to simplify the categorization of the data, each disease present in a fundus image is appended to the image's filename. This removes the need for further consulting the .csv file where the image labels are contained. 

3. **Image enhancement:** Each fundus image is enhanced through a Contrast Limited Adaptive Histogram Equalization (CLAHE) and (HSV) filter to increase the clarity of the image. 

4. **Left and right fundus Image fusion (optional):**  each left and right fundus image may be fused together into a single image by horizontally concatenating them or by performing an element-wise sum. After performing the fusion, the old left and right fundus images are removed from the dataset, so that by the end of this step only fused images remain. 

5. **Creation of validation set:**  30% of the training images are removed from the training set and placed into the validation set. 

6. **Image resizing:**  Each image is resized to 250 X 250 pixels (or 500 X 250, if image concatenation is used as an image fusion method). This allows the model to train against the data at a faster rate, while retaining the important features of the original data. 

#### Image enhancement

CLAHE and HSV are always applied to each image, though their parameters randomly vary within a defined range of values.  

Below is an example of a non-enhanced image (left) and the same image after enhancement. 

#### Image fusion

Two image fusion techniques are implemented: element-wise sum and concatenation.

Before the left and right fundus images may be fused together, they are resized to ensure that both images are of the same size. 

- **Element-wise sum** is performed by summing each pixel from the left fundus image with the corresponding pixel from the right fundus image. Images are represented as Numpy arrays in OpenCV, and summing two Numpy results in an element-wise sum. 

- **(Horizontal) image concatenation** is performed by using OpenCV's `hconcat` method. 

### Augmentation

After the data has been pre-processed, it is augmented to create new data from existing data. The augmentation techniques used are: horizontal and vertical flipping, random brightness increase and rotation (unless image concatenation is used as the image fusion method). Rotation is not applied to images which have been fused through concatenation, as they have a rectangular aspect ratio and rotating them would ruin the horizontal allignment of both fundus', creating an image that is not uniform with the rest of the data.  

### Models

## Results

## Analysis of Results

## How to run our model

## References

- [1] Li, Ning, Tao Li, Chunyu Hu, Kai Wang, and Hong Kang. “A Benchmark of Ocular Disease Intelligent Recognition: One Shot for Multi-Disease Detection.” Benchmarking, Measuring, and Optimizing,
  2021, 177–93. [https://doi.org/10.1007/978-3-030-71058-3_11.](https://doi.org/10.1007/978-3-030-71058-3_11)

- [2] Meller, Grzegorz. “AlgorithmsForMassiveData” Github. Accessed October 3rd, 2021. [https://towardsdatascience.com/ocular-disease-recognition-using-convolutional-neural-networks-c04d63a7a2da.](https://towardsdatascience.com/ocular-disease-recognition-using-convolutional-neural-networks-c04d63a7a2da)

- [3] JordiCorbilla. “Jordicorbilla/Ocular-Disease-Intelligent-Recognition-Deep-Learning: Odir-2019. Ocular Disease Intelligent Recognition through Deep Learning Architectures.” GitHub. Accessed October 3rd, 2021.
  [https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning.](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning)
