#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 18:01:51 2026

@author: hounsousamuel
"""
import os, sys
sys.path.insert(1, os.path.dirname(os.path.abspath(os.path.join(__file__, ".."))))
import aiofiles
import asyncio
import easyocr
import pdfplumber
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Request, Depends, HTTPException, status, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from classification_doc_avec_interface_react.config import model_params, IP, PORT, limite, REFUSED_EXT
from classification_doc_avec_interface_react.main_model import GB, DB
from classification_doc_avec_interface_react.chiffrement import FernetManager
from classification_doc_avec_interface_react.limiter import limiter
from classification_doc_avec_interface_react.utils import _get_file_name_mode, extract_text, rm, checksalt
from classification_doc_avec_interface_react.login_router import verify_username, update_num_analyzed

Baerer = HTTPBearer()
_model = None
_num = 0

router = APIRouter()

class Options(BaseModel):
    keep:int = 0
    true_label:str = ""
    username:str = ""


async def predict(filename, keep, true_label:str="", username:str="", ext:str=""):
    content = await extract_text(filename)
    pred = {
        "predict": "unknown",
        "proba": 1.0
        }
    if content:
        model = GB._get_classifier()
        pred = await asyncio.to_thread(model.predict, X=[content], label=true_label, name=username)
        
    else:
        print("Content vide !")
        return {
            "predict": "unknown",
            "proba": 1.0,
            }
    
    if not keep:
        await asyncio.to_thread(rm, filename)
    
    else:
        df = pd.DataFrame([{"textes": content, "label": true_label}])
        await asyncio.to_thread(df.to_csv, filename.replace(ext, ".csv"), index=False)
    print(pred)
    
    return pred
        
    
async def _analyse_one_file(file: UploadFile = File(...), keep:int = 0, true_label:str=""):
    keep = bool(keep)
    to_return = {
        "filename": file.filename,
        "content_type": file.content_type,
        "predict": {
            "predict": "unknown",
            "proba": 1.0,
            },
        "accepted": True
        }
    ext = os.path.splitext(file.filename)[-1]
    to_return["accepted"] = ext not in REFUSED_EXT
    filename = _get_file_name_mode(ext)
    ctn = await file.read()
    if not to_return['accepted']:
        return to_return
    
    try:
        async with aiofiles.open(filename, "wb") as f:
            await f.write(ctn)
    except:
        await asyncio.to_thread(rm, filename)
        to_return["accepted"] = False
        return to_return
    
    size = len(ctn) / (1024 * 1024)
    if size > 100:
        await asyncio.to_thread(rm, filename)
        raise HTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE,
            detail=f"Fichier trop lourd, {size} > 100 Mb)"
            )
        return to_return
    
    pred = await predict(filename, keep, true_label, ext)
    to_return['predict'] = pred  
    return to_return
        


@router.post("/analyze")
@limiter.limit(f"{limite}/minute")
async def _analyse(
    request: Request, 
    username: str = Form(...),
    true_label:str = Form(""),
    keep: int = Form(0),
    files: list[UploadFile] = File(...), 
    credentials:HTTPAuthorizationCredentials = Depends(Baerer)
    ):
    
    print(f"Nombre de fichiers reçus: {len(files)}") 
    username = username.strip()
    keep = bool(int(keep))
    true_label = true_label.strip()
    can = verify_username(username, credentials.credentials)
    if not can:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Non autorisé"
        ) 
    try:
        update_num_analyzed(name=username, to_add=len(files))
    except Exception as e:
        print('Erreur dans la mise à jour de num_analyzed : ', str(e))
        
    tasks = []
    for file in files:
        print(f"Fichier: {file.filename}, Type: {file.content_type}")
        tasks.append(asyncio.create_task(_analyse_one_file(file, keep, true_label)))
    results = await asyncio.gather(*tasks)
    print(results)
    return {"results": results}
    

        