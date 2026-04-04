#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 18:02:58 2026

@author: hounsousamuel
"""
import uuid
import os

IP = "0.0.0.0"
PORT = 9000
uuid_ = uuid.uuid4()

text_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "textes")
os.makedirs(text_dir, exist_ok=True)

model_params = {
    "lr": 0.01,
    "rs": 0,
    "cv": 5, 
    "units": [128, 64, 32, 16],
    "vocab_size": 30000,
    "model_name": "model_ee6aba19-87c2-4135-9234-f4c2b5225a13.pkl",
    }
limite = 10
ROUNDS = 100
# print(uuid_)
EXPIRE_TIME = 30
NBF = 2
REFUSED_EXT = [
    ".csv", ".rar", ".zip", ".xls", ".gz", ".tar", ".gz.tar", ".tar.gz"
    ]

MODEL_PATH = {
    "cat_dog": {
        "model_name": "model_binaire_cat_dog.keras",
        "class_name": "classes_model_binaire_cat_dog.json"
        },
    
    "malaria": {
        "model_name": "model_binaire_malaria.keras",
        "class_name": "classes_model_binaire_malaria.json"
        }
    }