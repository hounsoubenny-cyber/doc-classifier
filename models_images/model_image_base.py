#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 10:09:50 2026

@author: hounsousamuel
"""

import os, sys
import json
import joblib
import numpy as np
import uuid
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Dropout, MaxPooling2D, Flatten, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, hamming_loss, jaccard_score

_id = uuid.uuid4()

model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "model", "tf", "images")
os.makedirs(model_dir, exist_ok=True)

class_names_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "classes_names")
os.makedirs(class_names_dir, exist_ok=True)

class ImageDetection:
    def __init__(
            self,
            units:list=[],
            dropout:float=0.3,
            output_units=1,
            lr:float=1e-3,
            verbose:bool=True,
            class_mode="binary",
            model_name:str=f"model_img_{_id}.keras",
            class_name:str=f"classes_model_{_id}.json",
        ):
        self.units = units or [16, 32, 64, 128, 258]
        self.dropout = dropout or 0.3
        self.output_units = output_units
        self.class_mode = class_mode
        self.v = verbose
        self.lr = lr or 1e-3
        self.class_name = class_name
        self.model_name = model_name
        self.class_name_path = str(os.path.join(class_names_dir, self.class_mode))
        os.makedirs(self.class_name_path, exist_ok=True)
        self.class_name_path = str(os.path.join(self.class_name_path, class_name))
        self.model  = None
        self.model_path = str(os.path.join(model_dir, self.class_mode, model_name))
        self.size = (224, 224) # (299, 299)
    
    def _save(self, filename:str, value, mode:str='bin'):
        try:
            if mode.lower().strip() =="bin":
                joblib.dump(value, filename)
                print('Sauvegarde reussi dans avec joblib', filename)
            else:
                with open(filename, "w") as f:
                    json.dump(value, f, indent=2, ensure_ascii=False)
                    print('Sauvegarde reussi dans avec json', filename)
        except Exception as e:
            print("Erreur lors de la sauvegarde dans : ", filename, "\nErreur : ", str(e))
    
    def _load_model(self, path):
        return load_model(path)
    
    def _load_class_name(self, path:str):
        try:
            if path and os.path.exists(path):
                ext = os.path.splitext(path)[-1]
                if ext.lower() in (".pkl", ".joblib"):
                    return joblib.load(path)
                else:
                    with open(path, "r") as f:
                        return json.load(f)
        except Exception as e:
            print('Erreur dans le chargement du fichier des classes : ', str(e))
    
    def _get_model(self):
        models = []
        models.append(Input(shape=(self.size[0], self.size[1], 3,)))
        for unit in self.units:
            model = Conv2D(filters=unit, padding='same', kernel_size=(3, 3), activation="swish")
            models.append(model)
            models.append(MaxPooling2D(2, 2))
            # models.append(BatchNormalization())
            models.append(Dropout(self.dropout))
            
        models.append(Flatten())
        
        for unit in [128, 64, 32,]:   #16
            models.append(Dense(unit, activation="swish"))
            models.append(BatchNormalization())
            models.append(Dropout(self.dropout))
        
        models.append(Dense(self.output_units, activation="sigmoid" if (self.class_mode == "binary" and self.output_units == 1) else "softmax"))
        model = Sequential(models)
        model.compile(optimizer=Adam(self.lr), metrics=['accuracy', 'precision', "recall"], loss="binary_crossentropy" if self.output_units == 1 else "categorical_crossentropy")
        return model
    
    def _get_img_data_gen(self, what:str="fit", val_frac:float=0.2):
        if what == "predict":
            img_data_gen = ImageDataGenerator(rescale=1/255)
        else:
            img_data_gen = ImageDataGenerator(
                rescale=1/255,
                rotation_range=20,
                zoom_range=0.2,
                horizontal_flip=True,
                validation_split=val_frac,
                )
        return img_data_gen
    
    def _get_callbacks(self, monitor:str="val_loss", patience:int=20, mode="max"):
        return [
                EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True, verbose=self.v),
                ModelCheckpoint(self.model_path, monitor=monitor, save_best_only=True, verbose=self.v, mode=mode)
            ]
    
    def fit(
            self,
            directory:str, 
            val_frac:float=0.2, 
            batch_size:int=32, 
            epochs:int=20, 
            monitor:str="val_accuracy",
            mode:str="max",
            patience:int=20
            
        ):
        if not directory:
            raise ValueError('Répertoire requis !')
        model = self._get_model()
        img_data_gen = self._get_img_data_gen("fit", val_frac=val_frac)
        train_data = img_data_gen.flow_from_directory(
            directory,
            class_mode=self.class_mode,
            subset="training",
            target_size=self.size,
            batch_size=batch_size,
            shuffle=True,
            )
        
        val_data = img_data_gen.flow_from_directory(
            directory,
            class_mode=self.class_mode,
            subset="validation",
            target_size=self.size,
            batch_size=batch_size,
            shuffle=True,
            )
        print(train_data.class_indices, val_data.class_indices)
        # input()
        model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=self._get_callbacks(monitor=monitor, mode=mode, patience=patience)
            )
        self.model = model
        self.model.save(self.model_path.replace(".keras", "_all.keras"))
        class_names = train_data.class_indices
        to_save = {
            "class_num": class_names,
            "num_class": {v:k for k, v in class_names.items()}
            }
        self._save(self.class_name_path.replace('.json', ".pkl"), to_save, mode="bin")
        self._save(self.class_name_path, to_save, mode="json")
        # self._evaluate_model(self.model, val_data)
        
        return self
    
    # def _evaluate(self, model, val_data, class_mode="binary"):
    #     preds = np.array(model.predict(val_data))
    #     predict = np.asarray((preds > 0.5)).astype(int)
    #     cr = classification_report()
            
    def predict_binaire(self, path:str):
        if not self.model:
            self.model = self._load_model(self.model_path)
            if not self.model:
                raise ValueError('Model non disponible!  !')
        img_data_gen = self._get_img_data_gen(what="predict")
        data = img_data_gen.flow_from_directory(
            path,
            class_mode=None,
            shuffle=False,
            batch_size=1,
            target_size=self.size,
            )
        filenames = list(data.filenames)
        class_names = self._load_class_name(self.class_name_path.replace('.json', ".pkl"))
        # class_num = class_names.get('class_num')
        num_class = class_names.get('num_class')
        li = sorted(num_class.keys())
        dic = {k:num_class[k] for k in li}
            
        preds = self.model.predict(data)
        preds = np.array([[1 - p[0], p[0]] for p in preds])
        values = list(dic.values())
        to_return = {
            "proba": {k: dict(zip(values, [float(x) for x in preds[i]] )) for i, k in enumerate(filenames)},
            "predict": {k: dic[int(np.argmax(preds[i]))] for i, k in enumerate(filenames)}
            }
        # for i, k in enumerate(filenames):
        #     for_k = preds[i]
        #     to_return["proba"][k] = dict(zip(list(dic.values()), for_k))
        #     to_return['predict'][k] = dic[np.argmax(np.array(for_k))]
        
        return to_return
    
    def predict_categorical(self, path):
        if not self.model:
            self.model = self._load_model(self.model_path)
            if not self.model:
                raise ValueError('Model non disponible!  !')
        img_data_gen = self._get_img_data_gen(what="predict")
        data = img_data_gen.flow_from_directory(
            path,
            class_mode=None,
            shuffle=False,
            batch_size=1,
            target_size=self.size,
            )
        filenames = list(data.filenames)
        class_names = self._load_class_name(self.class_name_path.replace('.json', ".pkl"))
        # class_num = class_names.get('class_num')
        num_class = class_names.get('num_class')
        li = sorted(num_class.keys())
        dic = {k:num_class[k] for k in li}     
        preds = self.model.predict(data)
        values = list(dic.values())
        to_return = {
            "proba": {k: dict(zip(values, [float(m) for m in preds[i]] )) for i, k in enumerate(filenames)},
            "predict": {k: dic[int(np.argmax(preds[i]))] for i, k in enumerate(filenames)}
            }
        
        # for i, k in enumerate(filenames):
        #     for_k = preds[i]
        #     to_return["proba"][k] = dict(zip(list(dic.values()), for_k))
        #     to_return['predict'][k] = dic[np.argmax(np.array(for_k))]
        
        return to_return
        
        
if __name__ == "__main__":
    def test():
        img_detector = ImageDetection()
        img_detector._get_model()
        img_detector._get_img_data_gen()
        img_detector._get_callbacks()
    