#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 18:02:11 2026

@author: hounsousamuel
"""

import os, sys, json, pprint
import pandas as pd
import numpy as np
from model import Model
from config import model_params
dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', "textes")
os.makedirs(dataset_dir, exist_ok=True)

def fit(dataset_path:str, max_iter:int=10, opt:bool=False, func=pd.read_pickle):
    model = Model(**model_params)
    df = pd.DataFrame(func(dataset_path))
    X = df['textes']
    y = df['label']
    print(X)
    model.fit(X, y, optimize=opt, max_iter=max_iter)
    X_parts, y_part = X[:10], y[:10]
    preds = model.predict(X_parts)
    new_model = Model(**model_params)
    try:
        new_preds = new_model.predict(X)
    except:
        new_preds = {}
    
    try:
        print('Preds du model original : \n')
        print(json.dumps(preds, indent=2, ensure_ascii=False))
        print('Preds du nouveau model : \n')
        print(json.dumps(new_preds, indent=2, ensure_ascii=False))
    except:
        print('Preds du model original : \n')
        print(pprint.pprint(preds, indent=2))
        print('Preds du nouveau model : \n')
        print(pprint.pprint(new_preds, indent=2))
    
    for line  in X_parts.to_numpy():
        pred = model.predict([line])
        try:
            print(json.dumps(pred, indent=2, ensure_ascii=False))
        except:
            print(pprint.pprint(pred, indent=2))

if __name__ == "__main__":
    dataset_path = os.path.join(dataset_dir, "dataset.csv")
    fit(dataset_path, func=pd.read_csv)
