'''
Authors : Rich, Wu
Datetime: 2019/12/18
Describe: Use .h5 file demo vgg16's and resnet152's accuracy in different tree
'''




import os
from glob import glob
from keras import models, optimizers
import cv2
import numpy as np

DATASETS_PATH = "./datasets" 
CLASS_PATH = ["PCA", "feature"]

H5_FILES_PATH = ["./result/vgg16_", "./result/resnet152_"]


for cla in CLASS_PATH:
    
    TP_vgg = 0
    TP_resnet = 0
    ALL = 0
    vggModel = models.load_model(H5_FILES_PATH[0] + cla + ".h5")
    resnetModel = models.load_model(H5_FILES_PATH[1] + cla + ".h5")
    path = os.path.join(DATASETS_PATH, cla, "test")
    folderList = os.listdir(path)

    for folder in folderList:
        
        images = glob(os.path.join(path, folder, "*.jpg"))
        
        
        for image in images:

            img = cv2.imread(image)
           
            img = cv2.resize(img, (256, 256))
            img = np.reshape(img, (1, 256, 256, 3))

            vggPredict = vggModel.predict(img)[0].tolist()
            resnetPredict = resnetModel.predict(img)[0].tolist()
            
            vggIndex = vggPredict.index(max(vggPredict))
            resnetIndex = resnetPredict.index(max(resnetPredict))
            
            if folderList[vggIndex] == folder:
                TP_vgg += 1
            if folderList[resnetIndex] == folder:
                TP_resnet += 1
            
            ALL += 1
        
    
    print("{}: {:.2f}".format((H5_FILES_PATH[0] + cla), (TP_vgg/ ALL)))
    print("{}: {:.2f}".format((H5_FILES_PATH[1] + cla), (TP_resnet)))
