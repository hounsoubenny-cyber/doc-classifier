#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 19:24:01 2026

@author: hounsousamuel
"""

import os, sys
from jose import jwt, JWTError
from fastapi import HTTPException, status, Depends
from config import EXPIRE_TIME, NBF
from datetime import datetime, timedelta

def create_token(data:dict, key:bytes=b''):
    user = data.get('username', "")
    if not user:
        raise ValueError("Erreur dans création du token, username absent !")
    
    iat = datetime.utcnow()
    exp = iat + timedelta(minutes=EXPIRE_TIME)
    nbf = iat + timedelta(seconds=NBF)
    cop = {"sub": user, "exp": exp, "nbf": nbf, "iat": iat}
    token = jwt.encode(cop, key=key, algorithm="HS256")
    return token


def verify_token(token, key:bytes=b''):
    try:
        data = jwt.decode(token, key=key, algorithms=['HS256'])
        sub = data.get('sub', None)
        if sub :
            return sub
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Username introuvable !",
                headers={"WWW-Authenticate": "Baerer"}
                )
    
    except JWTError as e:
        print('Erreur jwt: ', type(e).__name__, ": ", str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide !",
            headers={"WWW-Authenticate": "Baerer"}
            )
    
    except Exception as e:
        print('Erreur : ', type(e).__name__, ": ", str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Erreur générale !",
            headers={"WWW-Authenticate": "Baerer"}
            )