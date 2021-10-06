# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 11:51:34 2021

@author: almamy
"""

import os
import numpy as np
from scipy.io import wavfile 
from hmmlearn import hmm #importing GaussianHMM 
import librosa # reading wavefilesfrom librosa.feature import mfcc #to extract mfcc features
import matplotlib.pyplot as plt
from scipy.io import wavfile
from librosa.feature import mfcc
import zipfile

#Chargement du fichier
def unzipfile(path):
  zip_ref = zipfile.ZipFile(path, 'r')
  zip_ref.extractall()
  zip_ref.close()
  
#Création du chemin et lecture du fichier
def loadfile(input_folder):
  for dirname in os.listdir(input_folder):
    subfolder = os.path.join(input_folder, dirname)
    label = subfolder[subfolder.rfind('/') + 1:]
    print(label)

#Entrainement du model
## Classe Gaussian HMMM
class HMMTrainer(object):
   def __init__(self, model_name='GaussianHMM', n_components=4):
     self.model_name = model_name
     self.n_components = n_components

     self.models = []
     if self.model_name == 'GaussianHMM':
        self.model=hmm.GaussianHMM(n_components=4)
     else:
        print("Please choose GaussianHMM")
   def train(self, X):
       self.models.append(self.model.fit(X))
   def get_score(self, input_data):
       return self.model.score(input_data)

##Extraction étiquette, obtention mffc et entrainement du modéle
def trainmodel(input_folder):

  # parcourir tous les fichiers de fruits et extraction de l'étiquette du fichier parent
  hmm_models = []
  for dirname in os.listdir(input_folder):
      subfolder = os.path.join(input_folder, dirname)
      if not os.path.isdir(subfolder): 
          continue
      label = subfolder[subfolder.rfind('/') + 1:]
      X = np.array([])
      y_words = []

      # obtention du mfcc de chaque fichier de
      # Read the input file
      for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
            filepath = os.path.join(subfolder, filename)
            sampling_freq, audio = librosa.load(filepath)            
            mfcc_features = mfcc(sampling_freq, audio)
            if len(X) == 0:
                X = mfcc_features[:,:15]
            else:
                X = np.append(X, mfcc_features[:,:15], axis=0)            
            y_words.append(label)
      #print('X.shape =', X.shape)

      # Entrainement du modéle
      hmm_trainer = HMMTrainer()
      hmm_trainer.train(X)
      hmm_models.append((hmm_trainer, label))
      hmm_trainer = None
  return hmm_models

#Prédiction sur les données de test
def prediction(input_files, hmm_models):
  for input_file in input_files:
        sampling_freq, audio = librosa.load(input_file)

          # Extract MFCC features
        mfcc_features = mfcc(sampling_freq, audio)
        mfcc_features=mfcc_features[:,:15]

        scores=[]
        for item in hmm_models:
            hmm_model, label = item
              
            score = hmm_model.get_score(mfcc_features)
            scores.append(score)
        index=np.array(scores).argmax()
        print("\nTrue:", input_file[input_file.find('/')+1:input_file.rfind('/')])
        print("Predicted:", hmm_models[index][1])