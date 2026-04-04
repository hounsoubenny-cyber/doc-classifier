#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 09:25:35 2026

@author: hounsousamuel
"""

import os, sys
import json, joblib

from sklearn.metrics import confusion_matrix, classification_report, hamming_loss, jaccard_score

def evaluate(path, class_path):
    data = None
    with open(path, "r") as f:
        data = json.load(f)
    
    class_names = joblib.load(class_path)
    class_dict = class_names['class_num']
    if not data or not class_dict:
        raise ValueError('Données vides !')
        
    pred = data["predict"]
    y_true = []
    y_pred = []
    for i, j in  class_dict.items():
        print(i, "  ---->  ", j)
        
    for k, v in pred.items():
        for i, j in class_dict.items():
            print(k, i, i in k)
            p = sum(c in k for c in i) / len(i)

            if i in k or p >= 0.85:
                y_true.append(j)
        y_pred.append(class_dict[v])
    
    print(len(y_pred), len(y_true))
    input()
    keys = list(class_dict.keys())
    report = classification_report(y_true, y_pred, labels=keys, target_names=keys)
    cm = confusion_matrix(y_true, y_pred, labels=keys)
    
    print('Confusion matrix : \n')
    print(cm)
    print("\nClassification report : \n", report)
    print(hamming_loss(y_true, y_pred))
    print(jaccard_score(y_true, y_pred, average="binary" if len(keys) == 2 else "weigthed"))

import json
import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score,
    precision_recall_fscore_support
)
import pandas as pd

def evaluate_predictions(json_file_path):
    """
    Évalue un modèle de classification d'images à partir des prédictions JSON
    """
    
    # Charger JSON
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    proba = data['proba']
    predictions = data['predict']
    
    # Extraire vraies classes depuis noms fichiers
    y_true = []
    y_pred = []
    y_proba = []
    filenames = []
    
    for filename, pred_class in predictions.items():
        # Extraire vraie classe depuis nom fichier (ex: "0_chat/image.jpg" -> "0_chat")
        true_class = filename.split('/')[0]
        
        y_true.append(true_class)
        y_pred.append(pred_class)
        
        # Récupérer probabilité prédiction
        proba_pred = proba[filename][pred_class]
        y_proba.append(proba_pred)
        filenames.append(filename)
    
    # Classes uniques
    classes = sorted(list(set(y_true)))
    print("="*60)
    print("📊 ÉVALUATION MODÈLE DE CLASSIFICATION")
    print("="*60)
    print(f"Classes détectées : {classes}")
    print(f"Nombre total d'images : {len(y_true)}\n")
    
    # ========== 1. MATRICE DE CONFUSION ==========
    print("="*60)
    print("1️⃣ MATRICE DE CONFUSION")
    print("="*60)
    
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Créer DataFrame pour affichage
    cm_df = pd.DataFrame(
        cm,
        index=[f"Vrai {c}" for c in classes],
        columns=[f"Prédit {c}" for c in classes]
    )
    print(cm_df)
    print()
    
    # Détails confusion matrix
    if len(classes) == 2:
        tn, fp, fn, tp = cm.ravel()
        print(f"✅ Vrais Positifs (TP)  : {tp}")
        print(f"✅ Vrais Négatifs (TN)  : {tn}")
        print(f"❌ Faux Positifs (FP)   : {fp}")
        print(f"❌ Faux Négatifs (FN)   : {fn}\n")
    
    # ========== 2. MÉTRIQUES PRINCIPALES ==========
    print("="*60)
    print("2️⃣ MÉTRIQUES DE PERFORMANCE")
    print("="*60)
    
    # Accuracy globale
    accuracy = accuracy_score(y_true, y_pred)
    print(f"🎯 ACCURACY GLOBALE : {accuracy:.2%}\n")
    
    # Rapport complet
    report = classification_report(
        y_true, 
        y_pred, 
        labels=classes,
        target_names=classes,
        output_dict=True
    )
    
    # Afficher par classe
    print("📊 MÉTRIQUES PAR CLASSE :")
    print("-" * 60)
    
    metrics_df = pd.DataFrame({
        'Classe': classes,
        'Precision': [report[c]['precision'] for c in classes],
        'Recall': [report[c]['recall'] for c in classes],
        'F1-Score': [report[c]['f1-score'] for c in classes],
        'Support': [report[c]['support'] for c in classes]
    })
    
    # Formater en pourcentages
    for col in ['Precision', 'Recall', 'F1-Score']:
        metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2%}")
    
    print(metrics_df.to_string(index=False))
    print()
    
    # Moyennes
    print("📈 MOYENNES :")
    print(f"   Macro Avg Precision : {report['macro avg']['precision']:.2%}")
    print(f"   Macro Avg Recall    : {report['macro avg']['recall']:.2%}")
    print(f"   Macro Avg F1-Score  : {report['macro avg']['f1-score']:.2%}\n")
    
    # ========== 3. ANALYSE DES ERREURS ==========
    print("="*60)
    print("3️⃣ ANALYSE DES ERREURS")
    print("="*60)
    
    errors = []
    for i, filename in enumerate(filenames):
        if y_true[i] != y_pred[i]:
            errors.append({
                'filename': filename,
                'true': y_true[i],
                'predicted': y_pred[i],
                'confidence': y_proba[i]
            })
    
    print(f"❌ Nombre total d'erreurs : {len(errors)} / {len(y_true)} ({len(errors)/len(y_true):.2%})\n")
    
    # Trier par confiance (pires erreurs = haute confiance mais faux)
    errors_sorted = sorted(errors, key=lambda x: x['confidence'], reverse=True)
    
    print("🔴 TOP 5 PIRES ERREURS (haute confiance mais faux) :")
    print("-" * 60)
    for i, err in enumerate(errors_sorted[:5], 1):
        print(f"{i}. {err['filename']}")
        print(f"   Vrai : {err['true']} | Prédit : {err['predicted']} ({err['confidence']:.2%} confiance)\n")
    
    # ========== 4. DISTRIBUTION PROBABILITÉS ==========
    print("="*60)
    print("4️⃣ DISTRIBUTION DES PROBABILITÉS")
    print("="*60)
    
    # Séparer prédictions correctes et incorrectes
    correct_probs = [y_proba[i] for i in range(len(y_true)) if y_true[i] == y_pred[i]]
    incorrect_probs = [y_proba[i] for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    
    print(f"✅ Prédictions CORRECTES :")
    print(f"   Moyenne confiance : {np.mean(correct_probs):.2%}")
    print(f"   Min : {np.min(correct_probs):.2%} | Max : {np.max(correct_probs):.2%}\n")
    
    if incorrect_probs:
        print(f"❌ Prédictions INCORRECTES :")
        print(f"   Moyenne confiance : {np.mean(incorrect_probs):.2%}")
        print(f"   Min : {np.min(incorrect_probs):.2%} | Max : {np.max(incorrect_probs):.2%}\n")
    
    # ========== 5. RECOMMANDATIONS ==========
    print("="*60)
    print("5️⃣ RECOMMANDATIONS")
    print("="*60)
    
    if accuracy >= 0.95:
        print("🎉 EXCELLENT ! Modèle très performant (>95% accuracy)")
    elif accuracy >= 0.85:
        print("✅ BON ! Modèle performant (85-95% accuracy)")
    elif accuracy >= 0.70:
        print("⚠️  MOYEN. Modèle acceptable mais améliorable (70-85% accuracy)")
    else:
        print("❌ FAIBLE. Modèle nécessite améliorations (<70% accuracy)")
    
    print()
    
    # Analyser par classe
    for classe in classes:
        prec = report[classe]['precision']
        rec = report[classe]['recall']
        
        if prec < 0.8 or rec < 0.8:
            print(f"⚠️  {classe} : Performance faible")
            if rec < prec:
                print(f"   → Recall faible ({rec:.2%}) : Modèle RATE beaucoup de {classe}")
                print(f"   → Solution : Ajouter plus d'exemples de {classe}")
            else:
                print(f"   → Precision faible ({prec:.2%}) : Modèle confond {classe} avec autres")
                print(f"   → Solution : Améliorer qualité données {classe}")
    
    print()
    
    # Confiance moyenne
    avg_confidence = np.mean(y_proba)
    if avg_confidence < 0.7:
        print("⚠️  Confiance moyenne faible (<70%) : Modèle hésitant")
        print("   → Solution : Plus d'epochs, meilleure architecture, ou plus de données")
    elif avg_confidence > 0.95:
        print("🎯 Confiance très élevée (>95%) : Modèle très sûr de lui")
        if accuracy < 0.90:
            print("   ⚠️  ATTENTION : Haute confiance mais erreurs = Possible overfitting")
    
    print("\n" + "="*60)
    print("✅ ÉVALUATION TERMINÉE")
    print("="*60)

    
    
if __name__ == "__main__":
    path, class_path = "data/preds_test/cat_dog/preds.json", "data/classes_names/binary/classes_model_binaire_cat_dog.pkl"
    # evaluate(path, class_path)
    evaluate_predictions(path)
        