import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import *
import os
import cv2
from sklearn.metrics import classification_report, confusion_matrix



class CRNN:
    
    def __init__(self,trainedPath=None,input_dim=(300, 129,3),no_classes=6):
        
        tf.random.set_seed(2)
        np.random.seed(2)
        
        self.trainedPath=trainedPath
        self.input_dim = input_dim
        self.no_classes = no_classes
        
        self._initializeBaseModel()
        
        
        
        
    def _initializeBaseModel(self):
        
        
        
        self.inpt = Input(shape=self.input_dim)

        self.conv1 = Conv2D(filters=16,kernel_size=7,activation="relu",padding="valid")
        self.batchNorm1 = BatchNormalization() 
        self.maxPool1 = MaxPooling2D(2,2)

        self.conv2 = Conv2D(filters=32,kernel_size=5,activation="relu",padding="valid")
        self.batchNorm2 = BatchNormalization()
        self.maxPool2 = MaxPooling2D(2,2)

        self.conv3 = Conv2D(filters=64,kernel_size=3,activation="relu",padding="valid")
        self.batchNorm3 = BatchNormalization()
        maxPool3 = MaxPooling2D(2,2)

        self.conv4 = Conv2D(filters=128,kernel_size=3,activation="relu",padding="valid")
        self.batchNorm4 = BatchNormalization()
        self.maxPool4 = MaxPooling2D(2,2)

        self.conv5 = Conv2D(filters=256,kernel_size=3,activation="relu",padding="valid")
        self.batchNorm5 = BatchNormalization()
        self.maxPool5 = MaxPooling2D(2,2)

        self.biLSTM = Bidirectional(LSTM(units = 256,return_sequences=True),input_shape=(14, 2048))

        self.flat = Flatten()
        self.dense1 = Dense(256)

        self.outp = Dense(self.no_classes,activation="softmax")
    
        
    
    def initializeModel(self,isBaseCnn=True,cnnModel="inceptionv3"):
        
        model = None
        
        if self.trainedPath is None:          
        
            if isBaseCnn:

                inp = self.conv1(self.inpt)
                x = self.batchNorm1(inp)
                x = self.maxPool1(x)
                print(x.shape)

                x = self.conv2(x)
                x = self.batchNorm2(x)
                x = self.maxPool2(x)
                print(x.shape)

                x = self.conv3(x)
                x = self.batchNorm3(x)
                x = self.maxPool3(x)
                print(x.shape)

                x = self.conv4(x)
                x = self.batchNorm4(x)
                x = self.maxPool4(x)
                print(x.shape)

                x = self.conv5(x)
                x = self.batchNorm5(x)
                x = self.maxPool5(x)
                print(x.shape)

                x = tf.squeeze(x,[2])
                x = self.biLSTM(x)[:,-1,:]
                print(x.shape)

                out = self.outp(x)
                print(out.shape)


                model = tf.keras.models.Model(inputs=inpt,outputs=out)
                model.summary()


            else:

                model = tf.keras.applications.inception_v3.InceptionV3(
                    input_shape=self.input_dim,
                    include_top=False,
                    classes=self.no_classes,
                )

                x = model(model.input)
                x = tf.math.reduce_mean(x,axis=2)
                print(x.shape)

                x = self.biLSTM(x)[:,-1,:]
                print(x.shape)

                out = self.outp(x)
                print(out.shape)

                model=tf.keras.models.Model(inputs=model.input,outputs=out)
                model.summary()

        else:
            
            model = tf.keras.models.load_model(self.trainedPath)
            model.summary()
            
        self.model = model
        
    
    def initializeData(self,trainPath,testPath,validationSplit=0.3):
        
        # create generator
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,validation_split=validationSplit
        )

        # prepare an iterators for each dataset
        self.train_it = datagen.flow_from_directory(
            directory=trainPath,
            class_mode='categorical',
            target_size=self.input_dim[:2],
        )
        self.valid_it = datagen.flow_from_directory(
            directory=trainPath,
            class_mode='categorical',
            target_size=self.input_dim[:2],
            subset='validation'
        )

        self.test_it = datagen.flow_from_directory(
            directory=testPath,
            class_mode='categorical',
            target_size=self.input_dim[:2],
        )

        # confirm the iterator works
        batchX, batchy = self.train_it.next()
        print('Train Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
        
        batchX, batchy = self.valid_it.next()
        print('Validation Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
        
        batchX, batchy = self.test_it.next()
        print('Test Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
        
        
        
    def train(self,epochs=10,savePath=None):
        
        
        
        
        self.model.compile(optimizer="adam",loss=tf.keras.losses.CategoricalCrossentropy(),metrics="accuracy")
        history = self.model.fit(self.train_it,validation_data=self.valid_it,epochs=epochs,workers = 20)
        
        if savePath is not None:
            self.model.save(f"{savePath}.h5")
        
        return history
    
    
    def test(self,isNew=False,newPath=None):
        
        test_it = self.test_it
        
        if isNew:
            
            # create generator
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255
            )

            test_it = datagen.flow_from_directory(
                directory=newPath,
                class_mode='categorical',
                target_size=self.input_dim[:2]
            )
        
        
        hist = self.model.evaluate(test_it)
        
        return hist
            
        
    
    
        
        
        
    
        
        
        
        
        