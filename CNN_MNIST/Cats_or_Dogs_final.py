# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 13:04:41 2019

@author: karm2204
"""
# make_main_dir = '../input'
# D:\input

import os, shutil
import pandas as pd
import numpy as np 
from PIL import Image
import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img
import random
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

from keras.models import Sequential
from keras.layers import Dense, Input, Flatten
from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


print(os.listdir("/input"))

#################################################################

# recursively merge two folders including subfolders just once 

#def mergefolders(root_src_dir, root_dst_dir):
#    for src_dir, dirs, files in os.walk(root_src_dir):
#        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
#        if not os.path.exists(dst_dir):
#            os.makedirs(dst_dir)
#        for file_ in files:
#            src_file = os.path.join(src_dir, file_)
#            dst_file = os.path.join(dst_dir, file_)
#            if os.path.exists(dst_file):
#                os.remove(dst_file)
#            shutil.copy(src_file, dst_dir)
#            
#mergefolders('D:/input/trainset/Cat', 'D:/input/trainset')
#mergefolders('D:/input/trainset/Dog', 'D:/input/trainset')
#shutil.rmtree('D:/input/trainset/Cat')
#shutil.rmtree('D:/input/trainset/Dog')
#os.remove('D:/input/trainset/.DS_Store')

#################################################################

IMAGE_WIDTH=64
IMAGE_HEIGHT=64
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3 # RGB color

filenames = os.listdir("D:/input/trainset")

categories = []
for filename in filenames:
    category = filename.split('.')[1]
#    if True:
#        print(category)
#        break
    if category == 'Dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

df.head()
df.tail()
df['category'].value_counts().plot.bar()


# see sample image

#import requests


sample = random.choice(filenames)
sample_image = load_img("D:/input/trainset/"+sample)
plt.imshow(sample_image)



# from sklearn.model_selection import train_test_split
# from tensorflow.keras.callbacks import TensorBoard
# from keras.layers.advanced_activations import LeakyReLU, PReLU

'''
# cropping images
import cv2
layers = [
     imageInputLayer([64 64 3],'Name','image')
     crop2dLayer('centercrop','Name','crop')
     ]
lgraph = layerGraph(layers)
lgraph = connectLayers(lgraph,'image','crop/ref')  
'''

# Initialising the CNN

'''
He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on 
imagenet classification." Proceedings of the IEEE international conference on computer 
vision. 2015.

classifier = Sequential()
act = keras.layers.advanced_activations.PReLU(init='zero', weights=None)
classifier.add(Dense(128, input_dim=14, init='uniform'))
# classifier.add(act)
'''
classifier = Sequential()
#classifier.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
classifier.add(Conv2D(32, (3, 3), activation='relu', input_shape=(3,64,64), data_format='channels_first'))
#classifier.add(BatchNormalization())

# pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# first convolutional layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D((2, 2), strides=2))
# second layer
classifier.add(Conv2D(128, (3, 3), activation='relu'))
classifier.add(MaxPooling2D((2, 2), strides=2))
# third layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D((2, 2)))
# flattening
classifier.add(Flatten())
# fully connected cnn
classifier.add(Dense(512, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))



classifier.compile(loss='binary_crossentropy',
                   optimizer='rmsprop', 
                   metrics=['accuracy'])

classifier.summary()


# To prevent over fitting stop the learning after 10 epochs and val_loss value not decreased

#earlystop = EarlyStopping(patience=10)

# reduce the learning rate if the accuracy not increase for 2 steps

#learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
#                                            patience=2, 
#                                            verbose=1, 
#                                            factor=0.5, 
#                                            min_lr=0.00001)

#callbacks = [earlystop, learning_rate_reduction]


# Prepare Test and Train Data

train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
train_df['category'].value_counts().plot.bar()
validate_df['category'].value_counts().plot.bar()

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=150



# All images will be rescaled by 1./255
# Traning Generator

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
        # This is the target directory
        train_df, "D:/input/trainset/",
        # All images will be resized to 150x150
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary',
        batch_size=batch_size
        )


# take a look at our augmented images:


#fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
#img_path = fnames[3]
#img = image.load_img(img_path, target_size=(150, 150))
#x = image.img_to_array(img)
## Reshape it to (1, 150, 150, 3)
#x = x.reshape((1,) + x.shape)


# Validation Generator

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(
        validate_df, 
        "D:/input/trainset/", 
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='binary',
        batch_size=batch_size
        )


#for data_batch, labels_batch in train_generator:
#    print('data batch shape:', data_batch.shape)
#    print('labels batch shape:', labels_batch.shape)
#    break

import tensorflow as tf

#epochs=3 if FAST_RUN else 10

history = classifier.fit_generator(
          train_generator,
          epochs = 100,
          validation_data=validation_generator,
          validation_steps=500,
          steps_per_epoch=100,
          )

classifier.save('cats_or_dogs_1.h5')


## Virtualize Training


#fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
#ax1.plot(history.history['loss'], color='b', label="Training loss")
#ax1.plot(history.history['val_loss'], color='r', label="validation loss")
#ax1.set_xticks(np.arange(1, epochs, 1))
#ax1.set_yticks(np.arange(0, 1, 0.1))
#
#ax2.plot(history.history['acc'], color='b', label="Training accuracy")
#ax2.plot(history.history['val_acc'], color='r',label="Validation accuracy")
#ax2.set_xticks(np.arange(1, epochs, 1))
#
#legend = plt.legend(loc='best', shadow=True)
#plt.tight_layout()
#plt.show()


# Prepare Testing Data

test_filenames = os.listdir("D:/input/testset/test")
test_df = pd.DataFrame({
    'filename': test_filenames
    })
nb_samples = test_df.shape[0]


#Create Testing Generator

test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "D:/input/testset/test/", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
    )


# Predict
# result return probability that image likely to be a dog.
predict = classifier.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))

# threshold 0.5 which mean if predicted value more than 50% it is a dog and under 50% will be a cat
threshold = 0.5
test_df['probability'] = predict
test_df['category'] = np.where(test_df['probability'] > threshold, 1,0)


# Virtaulize Result
test_df['category'].value_counts().plot.bar()


# predicted result with images

sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    probability = row['probability']
    img = load_img("D:/input/testset/test/"+filename, target_size=IMAGE_SIZE)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' '(' + "{}".format(round(probability, 2)) + ')')
plt.tight_layout()
plt.show()



# Submission output

submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('submission.csv', index=False)


#test_image = image.load_img('dataset/single_prediction/cat_or_dog.jpg', target_size = (img_width, img_height))
#test_image = image.img_to_array(test_image)
#test_image = np.expand_dims(test_image, axis = 0)
#result = classifier.predict(test_image )
#training_set.class_indices
#if result [0][0] == 1:
#    prediction = 'dog'
#else:
#    prediction = 'cat'

#%matplotlib inline

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()



# some optimizations

'''
import sklearn
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def classifier(optimizer):
    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), activation='relu',
                        input_shape=(img_width, img_height, 3)))
    # pooling
    classifier.add(MaxPooling2D((2, 2)))
    # first convolutional layer
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D((2, 2), strides=2))
    # second layer
    classifier.add(Conv2D(128, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D((2, 2), strides=2))
    # third layer
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D((2, 2)))
    # flattening
    classifier.add(layers.Flatten())
    # fully connected cnn
    classifier.add(Dense(512, activation='relu'))
    classifier.add(Dense(1, activation='sigmoid'))
    return classifier

classifier_optim = KerasClassifier(build_fn = classifier)
parameters = {'batch_size':[25, 32, 64], 
              'epock': [10, 20], 
              'optimizer': ['SGD','RMSprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
'''


submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('submission.csv', index=False)

