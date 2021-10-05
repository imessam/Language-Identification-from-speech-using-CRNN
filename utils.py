import numpy as np
import os
import shutil
import time
import sklearn
import librosa
from sklearn import preprocessing
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import IPython.display as ipd



def replaceZeroes(data):
    
    min_nonzero = np.min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    
    return data


def normalize(x, axis=0):
    
    """
    
    A technique used to adjust the volume of audio files to a standard set level; if this isn’t done, the volume can differ greatly from word to word, and the file can end up unable to be processed clearly.
    
    """
    
    
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

def preemphasis(x):
    
    """
    
        Pre-emphasis is done before starting with feature extraction. We do this by boosting only the signal’s high-frequency components, while leaving the low-frequency components in their original states. This is done in order to compensate the high-frequency section, which is suppressed naturally when humans make sounds.
        
    """
    
    
    return librosa.effects.preemphasis(x.astype(np.float64))




def playSound(audio,sr):
    
    ipd.display(ipd.Audio(audio,rate=sr))    


    
    
### The functions to extract spectograms from audio files was from this famous script https://www.frank-zalkow.de/en/create-audio-spectrograms-with-python.html
    

def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    
    """ 
    
    short time fourier transform of audio signal 
    
    """
    
    
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames) 



def logscale_spec(spec, sr=44100, factor=20.):
    
    """ 
    
    scale frequency axis logarithmically
    
    """    

    
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):        
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs




def extractSpectFromFile(audiopath,  plotpath=None,binsize=2**10, colormap="jet",isNormalize=False,isPreemphasis=False):
    
    """
    Plot spectrogram
    
    """
    samplerate, samples = wav.read(audiopath)
    
    if isNormalize:
        samples = normalize(samples)
    if isPreemphasis:
        samples=preemphasis(samples)
    
    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    
    
    try:
        ims = 20.*np.log10(replaceZeroes(np.abs(sshow)/10e-6)) # amplitude to decibel
    except:
        return

    timebins, freqbins = np.shape(ims)

    
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none");

    plt.xlim([0, timebins-1]);
    plt.ylim([0, freqbins]);

    xlocs = np.float32(np.linspace(0, timebins-1, 5));
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate]);
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)));
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs]);
    
    plt.axis('off');

    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight");
    else:
        plt.show();
        
    plt.cla()
    
    
    
    

def extractSpectFromFolder(path,outDir,compress=False):
    
    if os.path.isdir(outDir):
        shutil.rmtree(k)
        
    os.mkdir(outDir)
    
    print(f"Extracting from {path}")

    count = len(os.listdir(path))
    
    f=plt.figure(figsize=(15, 7.5));
    
    for file in os.listdir(path):
        
        audioFile = os.path.join(path,file).replace("\\","/")
        outFile = f"{outDir}/{file}.png"
        extractSpectFromFile(audioFile,plotpath=outFile)

        count -=1
        if count%100 == 0:
            print(f"Remaining {count}")

    plt.close(f)
    
    if compress:
        
        shutil.make_archive(outDir, 'zip', outDir)
        shutil.rmtree(outDir)
        
    
    

def extractSpectFromData(soundData = {},compress=False):
    
    """
    
        Extract audio spectograms for train and test data.
        
        input : dict with keys "train" and "test", and values for train and test paths.
    
    """
    

    for key1,paths in soundData.items():

        if os.path.isdir(key1):
            shutil.rmtree(key1)

        os.mkdir(key1)

        for key2,path in paths.items():

            start = time.time()

            outDir  = os.path.join(key1,key2).replace("\\","/")
            
            extractSpectFromFolder(path,outDir,compress)

            done = time.time()
            elapsed = done - start

            print(f"took : {round(elapsed,2)} seconds")

        if compress:
            
            shutil.make_archive(key1, 'zip', key1)
            shutil.rmtree(key1)

        
    
