#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 11:07:43 2026

@author: hounsousamuel
"""

import os, sys
from model_image_base import ImageDetection, _id

class ImageDetectionCategorical(ImageDetection):
    def __init__(
        self,
        units:list=[],
        dropout:float=0.3,
        output_units:int=1,
        lr:float=1e-3,
        verbose:bool=True,
        model_name:str=f"model_img_categorigal_{_id}.h5",
        class_name:str=f"classes_model_categorigal_{_id}.json",
    ):
        super().__init__(
            units=units,
            dropout=dropout,
            output_units=output_units,
            lr=lr,
            verbose=verbose,
            class_mode='categorical',
            model_name=model_name,
            class_name=class_name
            )
    
    def load_model(self, path:str):
        if os.path.exists(path):
            return self._load_model(path)
        else:
            raise ValueError('Chemin inexistant !')
    
    def predict(self, data_path:str):
        if os.path.exists(data_path):
            return self.predict_categorical(data_path)
        else:
            raise ValueError('Chemin inexistant !')
    
    