#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 18:02:03 2026

@author: hounsousamuel
"""

import os, sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import TextVectorization, Dense, LSTM, Conv1D, Embedding, Input, Dropout, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.models import Model as _Model
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler, LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report, confusion_matrix, hamming_loss, jaccard_score
from sklearn.utils.multiclass import type_of_target
from sklearn.feature_extraction.text import TfidfVectorizer
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
from sklearn.pipeline import Pipeline
import warnings
import dill
tf.random.set_seed(1)
pd.set_option("display.max_row", 111)
pd.set_option('display.max_columns', 111)
warnings.filterwarnings("ignore")

model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', "model")
os.makedirs(model_dir, exist_ok=True)

class KerasWrapper:
    def __init__(
            self, 
            y, 
            n_classes:int = None, 
            learning_rate:float=0.01, 
            vocab_size:int=30000, 
            units:list=[128, 64, 32, 16],
            seq_length:int=None
        ):
        y = np.asarray(y)
        self.seq_length = seq_length
        self.n_classes = n_classes or len(np.unique(y))
        self.lr = learning_rate
        self.y_shape = y.shape
        self.type_model = type_of_target(y).split("-")[0].lower()
        self.vocab_size = vocab_size
        self.units = units or [128, 64, 32, 16]
        self.min_unit = min(self.units)
    
    def __call__(self, meta:dict):
        n_classes_ = meta.get('n_classes_', self.n_classes)
        n_output, loss, activation = self._choice_good(n_classes_)
        inp = Input(shape=(self.seq_length,))
        x = Embedding(self.vocab_size, 256)(inp)
        x = Dense(128, activation="swish")(x)
        for unit in self.units:
            # x = Conv1D(filters=128, kernel_size=5, padding="same", activation="swish")(x)
            # x = Dropout(0.2)(x)
            # x = LayerNormalization()(x)
            # x = LSTM(128, activation="swish", return_sequences=bool(unit != self.min_unit))(x)
            # x = Dropout(0.2)(x)
            # x = LayerNormalization()(x)
            x = Dense(unit, activation="swish")(x)
            x = Dropout(0.2)(x)
            x = LayerNormalization()(x)
            
        x = GlobalAveragePooling1D()(x)
        out = Dense(n_output, activation=activation)(x)
        model = _Model(inp, out)
        model.compile(loss=loss, optimizer=Adam(self.lr), metrics=['accuracy', 'recall', 'precision'])
        self.model = model
        return model
    
    def _choice_good(self, n_classes: int):
        if self.type_model == "binary":
            return 1, "binary_crossentropy", "sigmoid"
        
        elif self.type_model == "multiclass":
            return n_classes, "categorical_crossentropy", "softmax"
        
        elif self.type_model == "multilabel":
            return n_classes, "binary_crossentropy", "sigmoid"
        
        else:
            raise ValueError(f'Type de y inconnue ou non adapté pour classification ({self.type_model})!')

class Model:
    def __init__(
            self, 
            lr:float = 0.01,
            rs:int=0,
            cv:int=5, 
            units:list=[128, 64, 32, 16],
            vocab_size:int=30000,
            model_name:str="model.pkl",
        ):
        self.lr = lr
        self.rs = rs
        self.vc = vocab_size
        self.cv = cv
        self.units = units
        self.model = None
        self.tv = None
        self.vocab_size = vocab_size
        self.model_path = os.path.join(model_dir, model_name)
        self.target_encoder = LabelEncoder()
        self.categories = [
            "facture", "contrat", "cv", "rapport", 
            "lettre", "formulaire", "reçu", "ordonnance", "autres",
        ]
        self.target_encoder.fit(self.categories)
        
    def save_model(self, to_save):
        try:
            with open(self.model_path, "wb") as f:
                dill.dump(to_save, f, recurse=True, protocol=4)
            
            print(f'Modèle sauvegardé avec succès dans {self.model_path}')
        except Exception as e:
            print(f'Erreur sauvegarde du modèle dans {self.model_path} : {str(e)}')
    
    def _load_model(self):
        try:
            with open(self.model_path, "rb") as f:
                data = dill.load(f)
            print(f'Modèle chargé avec succès, type de la donné chargé : {type(data)}')
            print('Chargement effectué avec succès depuis : ', self.model_path)
            return data
        except Exception as e:
            print(f'Erreur de chargement du modèle depuis {self.model_path} : {str(e)}')
            return e
    
    def load_model(self):
        data = self._load_model()
        self.model = data['model']
        self.tv = data["tv"]
        self.target_encoder = data['target_encoder']
        self.categories = data['cat']
        
    def _get_models(self, X, y):
        dic = {}
        xgb = XGBClassifier(
            n_estimators=5000,
            n_jobs=-1,
            objective="multi:softmax",
            random_state=self.rs,
            learning_rate=self.lr,
            tree_method='hist',
            )
        dic['xgb'] = ('xgb', xgb)
    
        hist = HistGradientBoostingClassifier(
            early_stopping=True,
            n_iter_no_change=100,
            max_iter=5000,
            learning_rate=self.lr,
            random_state=self.rs,
            class_weight="balanced",
            max_depth=None
            )
        
        dic['hist'] = ('hist', hist)
        
        rf = RandomForestClassifier(
            n_estimators=1000,
            random_state=self.rs,
            n_jobs=-1,
            max_depth=None,
            )
        dic['rf'] = ("rf", rf)
        
        logcv = LogisticRegressionCV(class_weight="balanced", cv=self.cv)
        dic['logcv'] = ('logcv', logcv)
        
        keras_model = KerasWrapper(y=y, learning_rate=self.lr, vocab_size=self.vc, units=self.units, seq_length=X.shape[1])
        def deep_func(meta:dict):
            return keras_model(meta)
        
        keras = KerasClassifier(
            model=deep_func, 
            validation_split=0.2,
            random_state=self.rs,
            class_weight="balanced",
            batch_size=32,
            epochs=64,
            callbacks=[EarlyStopping(patience=20)],
            loss="categorical_crossentropy",
            )
        # keras._estimator_type = "classifier"
        # dic['keras'] = ("keras", keras)
        
        return dic
        
    def _get_optimize_dict(self, name:str):
        PARAMS = {
            "hist": {
                "max_iter": Integer(4000, 8000),
                "learning_rate": Real(1e-4, 1e-1, prior="log-uniform"),
                "max_depth": Categorical([None, 10, 12, 14, 16])
                },
            
            "xgb": {
                "n_estimators": Integer(4000, 8000),
                "learning_rate": Real(1e-4, 1e-1, prior="log-uniform"),
                "max_depth": Categorical([None, 10, 12, 14, 16])
                },
            
            "rf": {
                "n_estimators": Integer(500, 2000),
                "max_depth": Categorical([None, 10, 12, 14, 16]),
                "max_features": Categorical(['sqrt', "log2"])
                },
            }
        return PARAMS.get(name.lower().strip(), {})
    
    def _optimize(self, dict_of_models:dict, X, y, max_iter:int = 10):
        excludes = ['keras', "logcv"]
        
        def _opt(dict_of_models:dict):
            best_models = {}
            for name, (_, model) in dict_of_models.items():
                print('Optimisation de : ', name.upper())
                start = time.time()
                space = self._get_optimize_dict(name)
                bayes = BayesSearchCV(
                    model,
                    search_spaces=space,
                    n_iter=max_iter,
                    n_jobs=-1,
                    scoring="precision",
                    return_train_score=True
                    )
                bayes.fit(X, y)
                best_models[name] = (name, bayes.best_estimator_)
                print(f'Fin optimisation en : {time.time() - start:.2f} secondes')
                print('Meilleur score cv : ', bayes.best_score_)
                print('Meillsur params : ', dict(bayes.best_params_))
                print('Infos d\'optimisations : \n', pd.DataFrame(bayes.cv_results_))
                print()
            return best_models
        
        models = {k:v for k, v in dict_of_models.items() if k not in excludes}
        best_models = _opt(models)
        
        for name in excludes:
            if name in dict_of_models:
                best_models[name] = (name, dict_of_models[name])
        
        return best_models
    
    def _get_tv(self, X):
        X = np.asarray(X)
        # max_seq_l = int(np.max([len(i) for i in X]))
        # min_seq_l = int(np.min([len(i) for i in X]))
        # print(max_seq_l)
        # max_seq_l = int(0.5 * (min_seq_l + max_seq_l))
        # tv = TextVectorization(self.vocab_size, output_mode="int", output_sequence_length=max_seq_l)
        # tv.adapt(X)
        # self.tv = tv
        tv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, max_features=self.vocab_size)
        tv.fit(X)
        self.tv = tv
        return self.tv
    
    def fit(self, X, y, max_iter:int=10, optimize:bool=False):
        X = np.asarray(X)
        y = np.asarray(y)
        self._get_tv(X)
        # X, y = np.asarray(self.tv(X)), self.target_encoder.transform(y)
        X, y = self.tv.transform(X).toarray(), self.target_encoder.transform(y)
        X_opt, _, y_opt, _ = tts(X, y, test_size=0.4)
        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
        models = self._get_models(X, y)
        if optimize:
            best_models = self._optimize(models, X_opt, y_opt, max_iter=max_iter)
        
        else:
            best_models = models
        
        meta = LogisticRegressionCV(class_weight="balanced", cv=self.cv)
        
        stack = StackingClassifier(
            estimators=list(best_models.values()),
            final_estimator=meta,
            passthrough=True,
            n_jobs=-1,
            cv=self.cv
            )
        pip = Pipeline([('rc', RobustScaler()), ('stack', stack)])
        pip.fit(X_train, y_train)
        self.model = pip
        self.evaluate(X_test, y_test)
        to_save = {
            "tv": self.tv,
            "model": self.model,
            "target_encoder": self.target_encoder,
            "cat": self.categories
            }
        self.save_model(to_save)
    
    def evaluate(self, X, y, average="binary"):
        print()
        print("="*20, "EVALUATION", "="*20)
        predict =  self.model.predict(X)
        try:
            cr = classification_report(y, predict)
            print('Classification report : \n', cr)
        except:
            pass
        
        try:
            cm = confusion_matrix(y, predict)
            print('Confusion matrix : \n', cm)
        except:
            pass
        
        try:
            score = self.model.score(X, y)
            print('Score : ', score)
        except:
            pass
        
        try:
            hml = hamming_loss(y, predict)
            print('hamming_loss(plus c\'est petit mieux c\'est) : ', hml)
        except:
            pass
        
        try:
            js = jaccard_score(y, predict, average=average)
            print('Jaccard score(plus c\'est grand mieux c\'est) : ', js)
        except:
            pass
        print("="*20, "FIN", "="*20)
        print()
    
    def predict(self, X):
        if not self.model:
            self._load_model()
            if not self.model:
                raise ValueError('Veuillez d\'abord entrainé le model avec la méthode fit !')
        X = np.asarray([X] if isinstance(X, str) else X)
        # X = np.asarray(self.tv(X))
        X = self.tv.transform(X).toarray()
        y_pred = np.asarray(self.model.predict(X)).astype(int)
        y_pred_proba = np.asarray(self.model.predict_proba(X)).astype(float)
        classes_ = [str(x) for x in self.target_encoder.classes_]
        try:
            classes_ = [float(x) for x in classes_]
        except:
            pass
        
        to_return = {
            "predict": {int(i):str(pred) for i, pred in enumerate(self.target_encoder.inverse_transform(y_pred))},
            "predict_proba": {i:dict(zip(classes_, [float(x) for x in line])) for i, line in enumerate(y_pred_proba)},
            }
        return to_return


    