# Image Classification
## Hand Gesture Recognition
This project is for Hand Gesture Recognition System. Convolutional Neural Network was used for feature extraction as well as classification of various hand gestures. Moreover, this program enables users to create their own hand gesture and register with a label name. Later on the new dataset can be re-trained using CNN. For training the Model only two convolutional layer was good enough.

### Creating Datasets
This project enables uesrs to create their own dataset for the application. From the menu, user can select the right option for creating new datasets. It will open open the WebCam and start capturing the image of hand gestures. After capturing the image data, the system needs to be re-trained. 

#### DataSet Directory
Dataset is stored in "dataset/" folder. Here, the each class datas are zipped. If anybody wants to use the dataset to train again, feel free to use them.

### Region Of Interest
At this moment, this project focuses on a particular region. If the hand is placed inside the ROI, then it will only be classified. I will make another version to make it more general.

### Data Processing
At first, each ROI is cropped from the video frames. Then the cropped image is converted into binary image so that only the hand becomes white and all the background becomes black. Later on this binary image (size=60 60) is feed through the CNN for feature extraction and classification.
#### ScreenShot of the Dataset
![](https://github.com/Nahid1992/ImageClassification--HandGestureRecognition_with_RegisterNewGesture_TFLearn/blob/master/ScreenShots/dataSetHandCount.png)

### Dataset Properties	
	Number of images for each class = 500
	Number of classes so far = 5
Having only 500 images per class gave around 99% of accuracy while training with Convolutional Neural Network

### Screen Shots of Performance Graph
#### Accuracy
![](https://github.com/Nahid1992/ImageClassification--HandGestureRecognition_with_RegisterNewGesture_TFLearn/blob/master/ScreenShots/Accuracy.png)
#### Loss
![](https://github.com/Nahid1992/ImageClassification--HandGestureRecognition_with_RegisterNewGesture_TFLearn/blob/master/ScreenShots/Loss.png)

### Screen Shots of the Application
-Coming Soon-

### Dependencies
* Python 3.6.2
* OpenCV 3.4.0
* TFLearn
* TensorBoard

