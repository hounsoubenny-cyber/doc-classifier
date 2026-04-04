#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 20:12:28 2026

@author: hounsousamuel
"""
import os, sys
import shutil
import pdfplumber
import easyocr
import asyncio
import uuid
import bcrypt
from fastapi import HTTPException, status
from config import REFUSED_EXT

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
text_dir = os.path.join(DIR, "fastapi_mount", "files")
os.makedirs(text_dir, exist_ok=True)

def _get_file_name_mode(ext:str):
    return os.path.join(text_dir, "filename_" + str(uuid.uuid4()) + ext)

def rm(path):
    try:
        if os.path.isfile(path):
            os.remove(path)  
        elif os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)  
        print(f"{path} supprmé avec succès !")
    except Exception as e:
        print(f"Erreur suppression {path}: {e}")

def _extract_pdf(filename:str):
    txt = ""
    with pdfplumber.open(filename) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                txt += page_text + "\n"
                
    return txt.strip()

def _extract_img(filename:str):
    reader = easyocr.Reader(['fr', 'en'])
    txt = ""
    results = reader.readtext(filename)
    for pos, text, prob in results:
        txt += text + " "
    return txt.strip()

def _extract_txt(filename:str):
    txt = ""
    with open(filename, "r") as f:
        txt = f.read()
    
    return txt.strip()

async def extract_text(filename:str):
    ext = os.path.splitext(filename)[-1]
    ext = str(ext).lower()
    txt = ""
    if ext in REFUSED_EXT:
        return txt
    
    if ext == ".pdf":
        txt = await asyncio.to_thread(_extract_pdf, filename)
    
    elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
        txt = await asyncio.to_thread(_extract_img, filename)
    
    else:
        try:
            txt = await asyncio.to_thread(_extract_txt, filename)
        except Exception as e:
            print(f"Erreur extraction {filename}: {e}")
    
    return txt

def checksalt(salt):
    try:
        pw = "password".encode()
        salt = salt.encode() if isinstance(salt, str) else salt
        bcrypt.hashpw(pw, salt)
        return True
    except ValueError:
        raise HTTPException(
            detail="salt invalide !",
            status_code=status.HTTP_406_NOT_ACCEPTABLE
            )