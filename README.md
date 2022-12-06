# Machine Learning Project

## Introduction
In our final project, we applied machine learning techniques to data from [the Cancer Imaging Archive](https://www.cancerimagingarchive.net/) (TCIA). Starting from three different pretrained convolutional neural network (CNN) image recognition models, we used a transfer learning process to train the models to distinguish computed tomography (CT) scans of lung tissue between lung cancer and and COVID-19 datasets, and to predict classification of new data between those two categories.

## Data Analysis and Visualization
The medical imaging data came in the DICOM format ([link](https://www.dicomstandard.org/)) which contains the arrays of pixel data as well as significant amounts of metadata about the interpretation of the data, as well as information about the imaging tests and the subject on which they were performed (de-identified so as to be anonymous prior to being made available to the public).

![DICOM components](https://github.com/crebello711/Project_4/blob/main/Resources/Images/medical_image_components.PNG)

These files can be read and viewed with the help of the [Insight Toolkit](https://itk.org/) (ITK), itkwidgets, and PyDICOM packages for Python. Each DICOM (.dcm) file representing a CT scan contains a 2dimensional array of pixels corresponding to a horizontal section of the portion of the subject scanned (lungs/chest in our case), and the series of these images corresponding to the entire scan can be put together as a 3dimensional visualization. See below for examples from itkwidgets.

![3d lungs](https://github.com/crebello711/Project_4/blob/main/Resources/Images/3d_lungs.PNG)
![3d lungs with section](https://github.com/crebello711/Project_4/blob/main/Resources/Images/3d_lungs_with_zplane.PNG)

## Data Preparation and Preprocessing
After identifying our two data sources
* [Lung cancer data](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70224216)
* [COVID-19 data](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=89096912)
from TCIA we had to find the datasets in the [Search Radiology Portal](https://nbia.cancerimagingarchive.net/nbia-search/) and then create shared "carts" by which to identify and access the data in our script using TCIA's Python utilities package

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
