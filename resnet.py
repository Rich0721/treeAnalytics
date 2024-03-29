'''
Authors : Rich, Wu
Datetime: 2019/12/18
Describe: Use resnet152 train can identify tree class.
'''

import os
import os.path as osp
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import ResNet152
from keras.layers import Flatten, Dense, Dropout
from keras import optimizers
from keras import backend as K
from keras.models import Model
from keras.callbacks import EarlyStopping
from time import time
import matplotlib.pyplot as plt
import numpy as np

INPUT_SHAPE = (256, 256, 3)
NUM_CLASSES = 3
FREEZE_LAYERS = 7

DATASET_PATH = "./datasets"
DATASETS_CLASS = ['PCA', 'feature']


# use ResNet152 model
def ResNet101_nn(train_path, vali_path, storageFileName):
    
    start_time = time()
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto',
            restore_best_weights=True)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
        )

    vali_datagen = ImageDataGenerator(
        rescale=1./255,
        )

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(256, 256),
        batch_size=8,
    )

    vail_generator = vali_datagen.flow_from_directory(
        vali_path,
        target_size=(256, 256),
        batch_size=8,
    )


    
    baseNet = ResNet152(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)

    net = baseNet.output
    net = Flatten()(net)
    net = Dropout(0.2)(net)
    net = Dense(NUM_CLASSES, activation='softmax')(net)

    ResNet = Model(inputs=baseNet.inputs, outputs=net)
    
    for layer in ResNet.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in ResNet.layers[FREEZE_LAYERS:]:
        layer.trainable = True
    
    ResNet.compile(optimizer=optimizers.RMSprop(lr=1e-5),
                    loss='categorical_crossentropy', metrics=['accuracy'])
    history = ResNet.fit_generator(
        train_generator,
        steps_per_epoch=50,
        epochs=30,
        validation_data=vail_generator,
        validation_steps=50,
        callbacks=[monitor]
    )

    ResNet.save(osp.join("result", "resnet152_" + storageFileName + "_3.h5"))
    plt_LineChart(history=history, netName="resnet152_" + storageFileName + ".jpg")
    end_time = time()
    mins = (end_time - start_time) // 60
    secs = (end_time - start_time) % 60
    print("ResNet Execute time: {}:{:.2f}".format(mins, secs))

def plt_LineChart(history=None, netName="test.jpg"):
    
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.plot(history.history["acc"], label="train_acc")
    plt.plot(history.history["val_acc"], label="val_acc")
    plt.title("{} Training Loss and Accuracy on sar classifier".format(netName[:-4]))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper right")
    plt.savefig(netName)


if __name__ == "__main__":

    for cla in DATASETS_CLASS:
        trainPath = osp.join(DATASET_PATH, cla, "train")
        valiPath = osp.join(DATASET_PATH, cla, "vali")

        ResNet101_nn(train_path=trainPath, vali_path=valiPath, storageFileName=cla)
    