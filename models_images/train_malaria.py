#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 11:08:17 2026

@author: hounsousamuel
"""

import os, sys
import json, pprint
from classification_binaire import ImageDetectionBinaire
from classification_categorical import ImageDetectionCategorical

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "binaire")
os.makedirs(data_dir, exist_ok=True)

malaria_dir = os.path.join(data_dir, "malaria")
os.makedirs(malaria_dir, exist_ok=True)
malaria_train_dir = os.path.join(malaria_dir, "train")
os.makedirs(malaria_train_dir, exist_ok=True)
malaria_test_dir = os.path.join(malaria_dir, "test")
os.makedirs(malaria_test_dir, exist_ok=True)

cat_dog_dir = os.path.join(data_dir, "cat_dog")
os.makedirs(cat_dog_dir, exist_ok=True)
cat_dog_train_dir = os.path.join(cat_dog_dir, "train")
os.makedirs(cat_dog_train_dir, exist_ok=True)
cat_dog_test_dir = os.path.join(cat_dog_dir, "test")
os.makedirs(cat_dog_test_dir, exist_ok=True)

def train_malaria():
    model = ImageDetectionBinaire(
        model_name="model_binaire_malaria.keras",
        class_name="classes_model_binaire_malaria.json"
        )
    
    model.fit(
        directory=malaria_dir,
        batch_size=32,
        monitor='recall'
        )
    preds = model.predict(
        data_path=malaria_test_dir
        )
    try:
        print(json.dumps(preds, indent=2, ensure_ascii=False))
    except:
        print(pprint.pprint(preds, indent=2))
        
    print(model)
    new_model = ImageDetectionBinaire(model_name=model.model_name, class_name=model.class_name)
    
    print(new_model.load_model(model.model_path))
    
    preds = new_model.predict(data_path=malaria_test_dir)
    try:
        print(json.dumps(preds, indent=2, ensure_ascii=False))
    except:
        print(pprint.pprint(preds, indent=2))
    
    return model

def train_cat_dog():
    model = ImageDetectionBinaire(
        model_name="model_binaire_cat_dog1.keras",
        class_name="classes_model_binaire_cat_dog1.json"
        )
    
    model.fit(
        directory=cat_dog_train_dir,
        batch_size=32,
        monitor='precision'
        )
    preds = model.predict(
        data_path=cat_dog_test_dir
        )
    try:
        print(json.dumps(preds, indent=2, ensure_ascii=False))
    except:
        print(pprint.pprint(preds, indent=2))
        
    print(model)
    new_model = ImageDetectionBinaire(model_name=model.model_name, class_name=model.class_name)
    
    print(new_model.load_model(model.model_path))
    
    preds = new_model.predict(data_path=malaria_test_dir)
    try:
        print(json.dumps(preds, indent=2, ensure_ascii=False))
    except:
        print(pprint.pprint(preds, indent=2))
    
    return model

def train2():
    model = ImageDetectionCategorical(
        model_name="model_binaire_test_chat_chien_autres_mock.keras",
        class_name="classes_model_cat.json",
        output_units=3
        )
    
    model.fit(
        directory="/home/hounsousamuel/PROJETS/classification_doc_avec_interface_react/models_images/data/categorical/chat_chien_autres/train",
        batch_size=16,
        monitor='recall'
        )
    preds = model.predict(
        data_path="/home/hounsousamuel/PROJETS/classification_doc_avec_interface_react/models_images/data/categorical/chat_chien_autres/train"
        )
    try:
        print(json.dumps(preds, indent=2, ensure_ascii=False))
    except:
        print(pprint.pprint(preds, indent=2))
        
    print(model)
    new_model = ImageDetectionCategorical(model_name=model.model_name, class_name=model.class_name, output_units=3)
    
    print(new_model.load_model(model.model_path))
    
    preds = new_model.predict(data_path=malaria_test_dir)
    try:
        print(json.dumps(preds, indent=2, ensure_ascii=False))
    except:
        print(pprint.pprint(preds, indent=2))
    
    
if __name__ == "__main__":
    # train_malaria()
    train_cat_dog()