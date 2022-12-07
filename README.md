# Machine Learning Project

## Introduction
In our final project, we applied machine learning techniques to data from [the Cancer Imaging Archive](https://www.cancerimagingarchive.net/) (TCIA). Starting from three different pretrained convolutional neural network (CNN) image recognition models, we used a transfer learning process to train the models to distinguish computed tomography (CT) scans of lung tissue between lung cancer and COVID-19 datasets, and to predict classification of new data between those two categories.

## Data Analysis and Visualization
The medical imaging data came in the DICOM format ([link](https://www.dicomstandard.org/)) which contains the arrays of pixel data as well as significant amounts of metadata about the interpretation of the data, information about the imaging tests, and the subject on which they were performed (de-identified so as to be anonymous prior to being made available to the public).

![DICOM components](https://github.com/crebello711/Project_4/blob/main/Resources/Images/medical_image_components.PNG)

These files can be read and viewed with the help of the [Insight Toolkit](https://itk.org/) (ITK), itkwidgets, and PyDICOM packages for Python. Each DICOM (.dcm) file representing a CT scan contains a 2dimensional array of pixels corresponding to a horizontal section of the portion of the subject scanned (lungs/chest in our case), and the series of these images corresponding to the entire scan can be put together as a 3dimensional visualization. See below for examples from itkwidgets.

![3d lungs](https://github.com/crebello711/Project_4/blob/main/Resources/Images/3d_lungs.PNG)
![3d lungs with section](https://github.com/crebello711/Project_4/blob/main/Resources/Images/3d_lungs_with_zplane.PNG)

## Data Preparation and Preprocessing
After identifying our two data sources
* [Lung cancer data](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70224216)
* [COVID-19 data](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=89096912)

from TCIA we had to find the datasets in the [Search Radiology Portal](https://nbia.cancerimagingarchive.net/nbia-search/) and then create shared "carts" by which to identify and access the data in our script using TCIA's Python utilities package ([link](https://github.com/kirbyju/TCIA_Notebooks/blob/main/tcia_utils.py)).
![getting shared cart](https://github.com/crebello711/Project_4/blob/main/Resources/Images/getting_shared_cart_name.PNG)

We created separate shared carts for each of the two datasets so that they would be downloaded into different folders which we could label to differentiate the datasets by filepath. In each case, we downloaded and read in the DICOM files from each cart, and then performed the same preprocessing steps on them before merging the datasets. 

First we had to remove all images that were not (512,512) pixels so that they were of a uniform size and could be put together into a 3 dimensional numpy array as the pretrained models we used require. Then we extracted the pixel data (a 2d numpy array) from each multi-component DICOM image file. The next step was to resize the pixel arrays to the format required for the models ((224,224) for VGG and ResNet, (299,299) for Inception).

Since these pretrained models expect full-color RGB images, with three color channels of data, but our image data from the CT scans was monochrome, we had to copy our data three times to simulate the three RGB channels, and used np.stack on each pixel array to do so. The final step at this stage was transposing the arrays from the (3,N,N) shape outputted by np.stack to the (N,N,3) shape required for input to the CNN models. 

Each 3d pixel array was appended to a list, while a list of target labels for the data (0 = covid data, 1 = cancer data) was created at the same time. These lists were converted to numpy arrays, which were concatenated to a single dataset and then passed through train_test_split.

Our final step of preprocessing was to run a preprocess_inputs algorithm specific to each pretrained model, which removes the mean (with respect to the "imagenet" dataset the models were pretrained on) from each color channel.

## Pretrained Models and Transfer Learning

To classify our CT image data, we chose to try three different pretrained image processing models: VGG19, ResNet50, and InceptionV3. These are deep convolutional neural net models that have been pretrained on the extremely large ImageNet dataset and come with initial sets of model weights that have already been optimized from that training.

Since we had our own target labels "cancer" and "covid" for the data, and did not want the models to try to classify predictions with the ImageNet labels (such, for example, as "nematode", "toilet seat", and "digital clock" that resulted from using the base predictions of the pretrained VGG19 on our data), we had to perform a transfer learning process in order to generalize these pretrained models.

This entailed taking the built-in output layer of the pretrained models, and adding a Flatten, a Dropout, and finally a Dense output layer that resulted in our desired binary classification. We set only the parameters of this final layer to be trainable. For our fitting step, we also added image augmentation preprocessing which applies random rotations, shifts, and flips to the input images before running them through the network.

## Results

Our initial runs were between 5 and 100 epochs but despite high accuracy scores the predictions on our testing data were only resulting in one label being predicted for all the test data. Increasing the number of epochs to 500 resulted in predictions that actually applied both target labels, so we kept with that number for the rest of our runs.

The bulk of our results came from running our transfer models with batch sizes of 32, with the Adam optimizer at its default learning rate = 0.001.

* ResNet50 Accuracy
    * ![resnet ct accuracy](https://github.com/crebello711/Project_4/blob/main/Resources/Images/resnet_ct_accuracy.png)
* ResNet50 Loss
    * ![resnet ct loss](https://github.com/crebello711/Project_4/blob/main/Resources/Images/resnet_ct_loss.png)
* VGG19 Accuracy
    * ![vgg ct accruacy](https://github.com/crebello711/Project_4/blob/main/Resources/Images/vgg_ct_accuracy.png)
* VGG19 ROC Curve
    * ![vgg roc curve](https://github.com/crebello711/Project_4/blob/main/Resources/Images/vgg_roc_curve.PNG)   
* InceptionV3 Accuracy
    * ![inception ct accuracy](https://github.com/crebello711/Project_4/blob/main/Resources/Images/inception_ct_accuracy.PNG)
* Inception V3 Loss
    * ![inception ct loss](https://github.com/crebello711/Project_4/blob/main/Resources/Images/inception_ct_loss.PNG)        

## Discussion

While all three models performed well in terms of accuracy on the datasets as we had initially processed them, we thought after our first successful runs that due to our COVID data initially coming from signed integer data, and the cancer data coming from unsigned integers, that the models may be simply detecting this trivial difference and not granting us any sort of medically useful classification abilities. 

We implemented rescalings of both datasets so their numerical values were in [0,1], and tried running our models again. However, now we observed the accuracy scores stopped improving after one or two epochs, with the losses varying slightly but not improving overall. This most likely means the models are getting trapped in a local minimum and failing to truly optimize. To overcome this, we tried:
* Varying the Adam optimizer learning rate between 0.001, 0.01, and 0.1.
* Changing to an SGD optimizer with a learning rate of 0.1 and momentum of 0.9.
* Varying the batch size between 100, 32, 2, and 1.

None of these steps succeeded in causing the models to not get immediately trapped in the local minimum. Using a batch size of 1 caused the accuracy score to vary slightly, but never show a trend of improvement even over hundreds of epochs.

We tried rescaling our data by hand, as well as with a StandardScaler function, and still the models were stuck.

![still stuck](https://github.com/crebello711/Project_4/blob/main/Resources/Images/still_stuck_after_standard_scaling.PNG)

---
## Initial Proposal
### For our final project, we will be investigating the problem of classifying computed tomography scans of lung tissue using machine learning. Our goal is to produce a script that creates, trains, and tests a machine learning model on medical imaging data from CT scans of lung tissue, predicting the classification of either lung cancer or COVID-19. 
Group Members: Douglas Packard, Chris Rebello, Karina Alvarez

### Data Source 1:
[Here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70224216) is our first data source. This website consists of CT and PET-CT DICOM images of lung cancer subjects.

### Data Source 2:
[Here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=89096912) is our second data source. This website contains patients who tested positive for COVID-19 along with their imaging (chest radiographs, chest CTs, brain MRIs, etc.).

---

- Data analysis and Cleaning: Python Pandas

- Data Storage: Amazon AWS, SQL Database

- Machine Learning: Scikit-learn, Tensorflow
