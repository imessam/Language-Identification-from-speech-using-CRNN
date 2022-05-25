import shutil
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm
from utils import *
from AudioSplitter import WavSplitter


model = tf.keras.models.load_model("trained models/CRNN InceptionV3 (300x129) original .h5")
splitter = WavSplitter()
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
            )

spectogsFolderName = "spectogos"
input_dim=(300, 129,3)
    
if os.path.isdir(spectogsFolderName):
    shutil.rmtree(spectogsFolderName)
os.mkdir(spectogsFolderName)


def run(path = "test"):
    
    folderName = path

    
    for file in tqdm(os.listdir(path)):
        
        outPath = splitter.multiple_split(folderName,file,4)
        outSpectsPath = extractSpectFromFolder(outPath,os.path.join(spectogsFolderName,file))
        
        test_it = datagen.flow_from_directory(
                    directory=spectogsFolderName,
                    class_mode=None,
                    target_size=input_dim[:2],
                )
        print("\nPredicting ....\n")
        preds = model.predict(test_it)
        preds = np.mean(preds,axis = 0)
        
        for i,score in enumerate(preds):
            print(f"{idx2label[i]} : {score}")
            
        pred = idx2label[np.argmax(preds)]
        print(f"\nPrediction is for {file} : {pred}\n")
        
        if os.path.isdir(outSpectsPath):
            shutil.rmtree(outSpectsPath)
            
    return

        
if __name__ == "__main__":
    
    if len(sys.argv)>1:
        run(sys.argv[1])
    else:
        run()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


