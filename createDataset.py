import numpy as np
import glob,os
import cv2
import time

class createDataset(object):
    def __init__(self):
        print('Register New DataSet -ON-')

    def register(self):
        self.CLASSNAME = input("Register ClassName = ")
        self.ImageSet = input("How many Image Set = ")
        self.LIST = self.webCamToggle()
        self.SaveImage()
        print('Dataset Created For -'+str(self.CLASSNAME)+'-')

    def webCamToggle(self):
        cam = cv2.VideoCapture(0)
        frameNumber = 0
        counter = -100
        limit_count = int(self.ImageSet)
        imageList = []
        saveToggle = False
        while counter<limit_count:
            if counter > -1:
                saveToggle = True
            frameNumber = frameNumber + 1
            ret,frame = cam.read()

            roi = frame[100:300,100:300]
            cv2.rectangle(frame,(100,100),(300,300),(0,0,255),0)

            roiGray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
            ret,binaryImg = cv2.threshold(roiGray,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            #im2, contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


            print( 'Frame Number = ' + str(frameNumber) )
            counter = counter + 1
            binaryImg = cv2.resize(binaryImg, (60,60))
            if saveToggle == True:
                cv2.putText(frame,'Image Saved = '+str(counter+1)+'/'+str(limit_count),(100,80), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255))
                imageList.append(binaryImg)
            cv2.imshow('WebCam',frame)
            cv2.imshow('Binary',binaryImg)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        cam.release()
        cv2.destroyAllWindows()
        return imageList

    def SaveImage(self):
        saveFolder = 'C:/Users/Nahid/Documents/MachineLearningProjects/#HandGestureRecognition/dataset'
        directory = saveFolder + '/' + str(self.CLASSNAME)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('Folder Created...')
        else:
            print('Could not create folder...')

        for index in range(0,len(self.LIST)):
            filename = directory + '/' + str(self.CLASSNAME) + '_'+str(index)+'.jpg'
            cv2.imwrite(filename,self.LIST[index])
            print('Image Saved = ' + str(index) + ':-> ' + filename)
        breakpoint=1
