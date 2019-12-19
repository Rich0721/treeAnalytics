'''
Authors : Rich, Wu
Datetime: 2019/12/19
Describe: Use .h5 file demo vgg16's and resnet152's accuracy in different tree
'''

import os
from glob import glob
from keras import models, optimizers
import cv2
import numpy as np
from time import time

DATASETS_PATH = "./datasets" 
CLASS_PATH = ["PCA", "feature"]

H5_FILES_PATH = ["./result/vgg16_", "./result/resnet152_"]


def demoTree(h5File=None, datasetPath=None):
    

    startTime = time()
    TP = 0
    ALL = 0

    model = models.load_model(h5File)
    folderList = os.listdir(os.path.join(datasetPath, "test"))
    
    for folder in folderList:
        
        images = glob(os.path.join(datasetPath, "test", folder,  "*.jpg"))
        
        for image in images:

            img = cv2.imread(image)
           
            img = cv2.resize(img, (256, 256))
            img = np.reshape(img, (1, 256, 256, 3))

            predict = model.predict(img)[0].tolist()
            
            
            maxIndex = predict.index(max(predict))
            
            if folderList[maxIndex] == folder:
                TP += 1
            ALL += 1

    endTime = time()
    mins = int((endTime - startTime) // 60)
    secs = int((endTime - startTime) % 60)
    
    print("{} execute time: {}:{}".format(h5File[:-3], mins, secs))
    print("{} accuracy: {:.2f}".format(h5File[:-3], (TP/ALL)))


if __name__ == "__main__":
    
    for cla in CLASS_PATH:
        for h5 in H5_FILES_PATH:
            datasetPath = os.path.join(DATASETS_PATH, cla)
            h5file = h5 + cla + "_3.h5"
            demoTree(h5File=h5file, datasetPath=datasetPath)