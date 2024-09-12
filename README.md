#IMPORTING PACKAGES AND CALLING THE DIRECTORY OF DATASET
import Numpy as np
import pandas as pd
import os
from re import search
import shutil
import natsort
from PIL import Image
import Matplotlib. pyplot as plt
import cv2
from tqdm import tqdm
DIR=r'images/Original dataset'
train=pd. read_csv(r"labels/train.csv")
test=pd. read_csv(r"labels/test.csv")
train. Head ()
PREPARE THE TRAINING DATA
class_names=train.loc[:,'healthy':]. columns
print(class_names)
number=0
train['label’] =0
for i in class_names:
    train['label’] =train['label'] + train[i] * number
    number=number+1
train. Head ()
DIR
natsort. natsorted (os. listdir(DIR))
def get_label_img(img):
    if search ("Train”, img):
        img=img. Split (‘.’) [0]
        label=train.loc[train['image_id’] ==img] ['label']
        return label
def create_train_data ():
    images=natsort. natsorted (os. listdir(DIR))
    for img in tqdm(images):
        label=get_label_img(img)
        path=os. path. Join (DIR, img)
        
        if search ("Train”, img):
            if (img. Split ("_”) [1]. split (“.”) [0]) and label. Item () ==0:
                shutil. Copy (path, r’images/train/healthy')
            
            elif (img. Split ("_”) [1]. split (“.”) [0]) and label. Item () ==1:
                shutil. Copy (path, r’images/train/alternaria_rot')
                
            elif (img. Split ("_”) [1]. split (“.”) [0]) and label. Item () ==2:
                shutil. Copy (path, r’images/train/aspergillus_rot')
                
            elif (img. Split ("_”) [1]. split (“.”) [0]) and label. Item () ==3:
                shutil. Copy (path, r’images/train/botrytris_fungus')
                
        elif search ("Test”, img):
            shutil. Copy (path, r’images/test')
                shutil. Os. mkdir(r'images/train')
shutil.os. mkdir(r'images/train/healthy')
shutil.os. mkdir(r'images/train/alternaria_rot')
shutil.os. mkdir(r'images/train/aspergillus_rot')
shutil.os. mkdir(r'images/train/botrytris_fungus')
shutil.os. mkdir(r'images/test')
train_dir=create_train_data ()
DATA PREPROCESSING
Train_DIR=r'images/train'
Categories=['healthy','alternaria_rot','aspergillus_rot','botrytris_fungus']

for j in Categories:
    path=os. path. Join (Train_DIR, j)
    for img in os. listdir(path):
        old_image=cv2.imread(os. path. Join (path, img), cv2.COLOR_BGR2RGB)
        plt. imshow(old_image)
        plt. show ()
        break
    break
IMG_SIZE=224
new_image=cv2.resize(old_image, (IMG_SIZE, IMG_SIZE))
plt. imshow(new_image)
plt. show ()
MODEL PREPARATION
import tensorflow as tf
from tensorflow. keras. Models import Sequential
from tensorflow. keras. Callbacks import ModelCheckpoint, EarlyStopping
from tensorflow. keras. preprocessing. Image import ImageDataGenerator
from tensorflow. keras. Layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
datagen=ImageDataGenerator (rescale=1. /255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                vertical_flip=True,
                                validation_split=0.2)


train_datagen=datagen. flow_from_directory (r'images/train',
                                         target_size= (IMG_SIZE, IMG_SIZE),
                                         batch_size=16,
                                         class_mode='categorical',
                                         subset='training')

val_datagen=datagen. flow_from_directory (r'images/train',
                                         target_size= (IMG_SIZE, IMG_SIZE),
                                         batch_size=16,
                                         class_mode='categorical',
                                         subset='validation')
model=Sequential ()
model.add (Conv2D (64, (3,3), activation='relu’, padding='same’, input shape= (IMG_SIZE, IMG_SIZE,3)))
model.add (MaxPooling2D (2,2))
model.add (Conv2D (64, (3,3), activation='relu’, padding='same'))
model.add (MaxPooling2D (2,2))
model.add (Conv2D (64, (3,3), activation='relu’, padding='same'))
model.add (MaxPooling2D (2,2))
model.add (Conv2D (128, (3,3), activation='relu’, padding='same'))
model.add (MaxPooling2D (2,2))
model.add (Flatten ())
model.add (Dense (4, activation='softmax'))

# Compile the Model
model. compile (optimizer=tf. keras. optimizers. Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
model. Summary ()
checkpoint=ModelCheckpoint (r'models\apple2.h5',
                          monitor='val_loss',
                          mode='min',
                          save_best_only=True,
                          verbose=1)
earlystop=EarlyStopping (monitor='val_loss',
                       min_delta=0,
                       patience=10,
                       verbose=1,
                       restore_best_weights=True)

callbacks= [checkpoint, earlystop]
model_history=model.fit_generator (train_datagen, validation_data=val_datagen,
                                 epochs=20,
                                 steps_per_epoch=train_datagen. samples//16,
                                 validation_steps=val_datagen. samples//16,
                                 callbacks=callbacks)
acc_train=model_history. history['accuracy']
acc_val=model_history. history['val_accuracy']
epochs=range (1,21)
plt. plot (epochs, acc_train,'g’, label='Training Accuracy')
plt. plot (epochs, acc_val,'b’, label='Validation Accuracy')
plt. Title ("Training and Validation Accuracy")
plt. xlabel("Epochs")
plt. ylabel("Accuracy")
plt. legend ()
plt. show ()
loss_train=model_history. history['loss']
loss_val=model_history. history['val_loss']
epochs=range (1,21)
plt. plot (epochs, loss_train,'g’, label='Training Loss')
plt. Plot (epochs, loss_val,'b’, label='Validation Loss')
plt. Title ("Training and Validation Loss")
plt. ylabel("Epochs")
plt. ylabel("Loss")
plt. Legend ()
plt. Show ()
MAKING THE PREDICTION ON A SINGLE IMAGE
test_image=r'images/train/alternaria_rot/Test_400'
test_image=r'prashanth_1.jpg'
image_result=Image. Open(test_image)
from tensorflow. keras. preprocessing import image
test_image=image. load_img (test_image, target_size= (224,224))
test_image=image.img_to_array(test_image)
test_image=test_image/255
test_image=np. expand_dims (test_image, axis=0)
result=model. Predict(test_image)
print (np. argmax(result))
Categories=['healthy','alternaria_rot','aspergillus_rot','botrytris_fungus']
image_result=plt. imshow(image_result)
plt. Title (Categories [np. argmax(result)])
plt. Show ()

