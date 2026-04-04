#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 21:15:44 2026

@author: hounsousamuel
"""

import os, sys
from config import model_params
from model import Model
from diskcache import Cache
from chiffrement import FernetManager
from database import Client, DBManager
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "var", "cache")
os.makedirs(cache_dir, exist_ok=True)
CACHE = Cache(cache_dir)
C_EXPIRE = 60 * 60
db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "databases")
os.makedirs(db_dir, exist_ok=True)

class Global:
    def __init__(self):
        self._classifier = None
    
    def _get_classifier(self):
        if self._classifier:
            return self._classifier
        
        else:
            self._classifier = Classsifier()
            return self._classifier
        
GB = Global()
DB = DBManager()

class Classsifier:
    def __init__(self):
        self.model = Model(**model_params)
        self.model.load_model()
    
    def _update_history(self, predicted, h_id, name:str = "", _id:int=None, _all:bool=False):
        to_set = {h_id: predicted}
        result = DB.update_history(class_name=Client, name=name, _id=_id, _all=_all, history=to_set)
        return result
    
    def _get_history(self, name:str="", _id:int=None, _all:bool=False):
        user = DB.get_user(_all=False, name=name, class_name=Client)
        if user:
            history = user.history
            return history
        
        else:
            return {}
    
    def _predict(self, X, label:str=""):
        h_id = hash(X if isinstance(X, str) else X[0])
        value = CACHE.get(h_id, None)
        if value:
            return value
        
        pred = self.model.predict([X] if isinstance(X, str) else X)
        predict = pred.get("predict", {}).get(0, "unknown")
        proba = pred.get('predict_proba', {}).get(0, {}).get(predict, 0.0)
        data = {
            "predict": predict,
            "proba": float(f"{proba:.2f}"),
            }
        
        meta = CACHE.get('metadata', {})
        if meta:
            count = meta.get('count', 0) + 1
            fea = meta.get('fea', []).append((X, label))
        else:
            count = 0
            fea = [(X, label)]
        CACHE.set('metadata', {"count": count, "features": fea})
        CACHE.set(h_id, data, expire=C_EXPIRE)
        return data
    
    def predict(self, X, label:str="", name:str=""):
        h_id = hash(X if isinstance(X, str) else X[0])
        history = self._get_history(name=name, _all=False)
        if history:
            value = history.get(h_id, None)
            if value:
                return value
                
        pred = self._predict(X, label)
        print(pred)
        r = self._update_history(predicted=pred, h_id=h_id, name=name, _all=False)
        print('Mise à jour historique : ', r)
        return pred
    
    def _refit(self):
        pass
    
    def refit_monitor(self):
        pass
    
    def refit_surveillance(self):
        pass


        
            
        