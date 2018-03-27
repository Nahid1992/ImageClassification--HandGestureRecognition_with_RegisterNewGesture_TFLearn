import cv2
import numpy as np
from cv2 import boundingRect, countNonZero, cvtColor, drawContours, findContours, getStructuringElement, imread, morphologyEx, pyrDown, rectangle, threshold
from createDataset import createDataset
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_utils import build_hdf5_image_dataset
from tflearn.layers.normalization import local_response_normalization
from glob import glob

def webCamToggle(model,true_list):
    cam = cv2.VideoCapture(0)
    frameNumber = 0
    while True:
        frameNumber = frameNumber + 1
        ret,frame = cam.read()

        roi = frame[100:300,100:300]
        cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)

        roiGray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        ret,binaryImg = cv2.threshold(roiGray,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #im2, contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        binaryImg = cv2.resize(binaryImg,(60,60))
        binaryImg = binaryImg.reshape(60,60,1)
        binaryImg = binaryImg/255
        predict_hand = model.predict([binaryImg])
        predict_hand = np.array(predict_hand)
        predict_gesture = true_list[predict_hand.argmax()]
        cv2.putText(frame,str(predict_gesture),(100,80), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,255))
        cv2.imshow('WebCam',frame)
        #cv2.imshow('Binary',binaryImg)
        print( 'Frame Number = ' + str(frameNumber) +': -> ' + predict_gesture +'->'+str(predict_hand))
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cam.release()
    cv2.destroyAllWindows()

    return "Thanks..."

def create_model(nb_classes):
    # Real-time data preprocessing
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Real-time data augmentation
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)

    network = input_data(shape=[None, 60, 60, 1],data_preprocessing=img_prep,data_augmentation=img_aug)

    network = conv_2d(network, 30, 3, strides=2, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = conv_2d(network, 30, 3, strides=2, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = fully_connected(network, 128, activation='relu')
    network = fully_connected(network, nb_classes, activation='softmax')
    model = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)

    return model

def train_model(X,Y):
    model = create_model(Y.shape[1])
    model = tflearn.DNN(model, tensorboard_verbose=3)
    model.fit(X, Y, n_epoch=10, validation_set=0.1, shuffle=True, show_metric=True, batch_size=100, snapshot_step=100, snapshot_epoch=False, run_id='HandTrack_run01')

	#save Model
    model.save('models/tflearn_handGesture_model.model')
    print('Model Saved...')

def main():
    while True:
        print('------------------------------------')
        print('1. Register New Class')
        print('2. Re-Train Model')
        print('3. Start Application')
        print('Quit')
        print('------------------------------------')
        task = input('Input = ')

        if task == '1':
            create_dataset = createDataset()
            create_dataset.register()
        elif task == '2':
            dataset_file = 'dataset'
            X,Y = tflearn.data_utils.image_preloader(dataset_file,image_shape=(60,60),mode='folder',categorical_labels=True,normalize=True)
            X,Y = tflearn.data_utils.shuffle(X,Y)
            X = X.reshape(-1,60,60,1)
            train_model(X,Y)
        elif task == '3':
            true_list = []
            file = glob('dataset/*')
            for i in range(0,len(file)):
                className = file[i].split("\\")[-1]
                true_list.append(className)
            model = create_model(len(true_list))
            model = tflearn.DNN(model)
            model.load('models/tflearn_handGesture_model.model')
            A = webCamToggle(model,true_list)
        else:
            break
    print('Completed...')

if __name__== "__main__":
  main()
