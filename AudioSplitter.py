from pydub import AudioSegment
import math
import os
from tqdm.auto import tqdm


class WavSplitter():
    
    def __init__(self):
        
        self.tempPath = "temp"
        
        if not os.path.isdir(self.tempPath):
            os.mkdir(self.tempPath)
                
    
    def get_duration(self,audio):
        return audio.duration_seconds
    
    def single_split(self, audio, from_sec, to_sec, outPath):
        t1 = from_sec * 1000
        t2 = to_sec * 1000
        split_audio = audio[t1:t2]
        split_audio.export(outPath, format="wav")
        
    def multiple_split(self,folderName,fileName,sec_per_split):
        
        outPath = os.path.join(self.tempPath,fileName)
        
        print(f"\nSplitting {fileName} into samples of 4 seconds each...\n")
        
        if not os.path.isdir(outPath):
            os.mkdir(outPath)
        
        filePath = os.path.join(folderName,fileName)
        
        audio = AudioSegment.from_wav(filePath)
        
        total_sec = math.ceil(self.get_duration(audio))
        for i in tqdm(range(0, total_sec, sec_per_split)):
            outFileName = str(i) + '_' + fileName
            self.single_split(audio, i, i+sec_per_split, os.path.join(outPath,outFileName))
            if i == total_sec - sec_per_split:
                print('\nAll splited successfully\n')
                
        return outPath
                
                
                
                