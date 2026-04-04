#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 05:20:04 2026

@author: hounsousamuel
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import tensorflow as tf
from config import MODEL_PATH
from classification_binaire import ImageDetectionBinaire, ImageDetection
from classification_categorical import ImageDetectionCategorical
from diskcache import Cache

cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "var", "cache")
os.makedirs(cache_dir, exist_ok=True)
CACHE = Cache(cache_dir)
C_EXPIRE = 60 * 60

class ModelImageDetection:
    def __init__(self):
        self.cat_dog_model = None
        self.malaria_model = None
        self.cancer_sein_model = None
    
    def _predict(self, data_path:str, model_name:str):
        model_name = model_name.strip().lower()
        preds = {}
        if "malaria" in model_name:
            if not self.malaria_model:
                model_name = MODEL_PATH.get('malaria', {}).get('model_name', "")
                class_name = MODEL_PATH.get('malaria', {}).get('class_name', "")
                if not (model_name and class_name):
                    raise ValueError('Configuration mal faites, model_name et class_name sont absents !')
                self.malaria_model = ImageDetectionBinaire(model_name=model_name, class_name=class_name)
                preds = self.malaria_model.predict(data_path=data_path)
        
        elif "cat_dog" in model_name:
            if not self.cat_dog_model:
                model_name = MODEL_PATH.get('cat_dog', {}).get('model_name', "")
                class_name = MODEL_PATH.get('cat_dog', {}).get('class_name', "")
                if not (model_name and class_name):
                    raise ValueError('Configuration mal faites, model_name et class_name sont absents !')
                self.cat_dog_model = ImageDetectionBinaire(model_name=model_name, class_name=class_name)
                preds = self.cat_dog_model.predict(data_path=data_path)
        
        elif "cancer_sein" in model_name:
            if not self.cancer_sein_model:
                model_name = MODEL_PATH.get('cancer_sein', {}).get('model_name', "")
                class_name = MODEL_PATH.get('cancer_sein', {}).get('class_name', "")
                if not (model_name and class_name):
                    raise ValueError('Configuration mal faites, model_name et class_name sont absents !')
                self.cancer_sein_model = ImageDetectionBinaire(model_name=model_name, class_name=class_name)
                preds =  self.cancer_sein_model.predict(data_path=data_path)
        
        else:
            raise ValueError('Model name non pris en charge !')
        
        return preds
        
            