#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 20:13:31 2026

@author: hounsousamuel
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fastapi import APIRouter, status, HTTPException, Request, Depends, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from auth_jwt import verify_token, create_token
from chiffrement import hashpw, checkpw
from limiter import limiter
from config import limite, IP, PORT, model_params
from main_model import DB, Client, FernetManager
from classification_doc_avec_interface_react.utils import checksalt
from diskcache import Cache
import cryptography

USERS = Cache(".user_cache")
TTL = 3600 * 3
login_router = APIRouter()
baerer = HTTPBearer()

class UserInfo(BaseModel):
    name: str
    password: str
    age: int
    contact: str = ""
    salt: str = ""
    

def verify_username(username, token):
    if username in USERS:
        try:
            sub = verify_token(token=token, key=USERS.get(username))
        except Exception as e:
            raise e
        if sub != username:
            raise HTTPException(
                status_code=status.HTTP_406_NOT_ACCEPTABLE,
                detail="Le propriétaire du token n'est pas celui qui fait la demande"
                )
        return True
    else:
        raise HTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE,
            detail="Username inconnue, veuillez vous authentifier d'abord"
            )
        
def update_num_analyzed(name, to_add:int=1):
    user = DB.get_user(class_name=Client, _all=False, name=name)
    if user and name in USERS:
        salt = USERS.get(name)
        fernet = FernetManager(salt=salt, password=user.password)
        num = user.num_analyzed
        if isinstance(num, bytes):
            num = fernet.decrypt(num)
        if num:
            num = int(num) + to_add
        else:
            num = to_add
        num = fernet.encrypt(num)
        s = DB.update_num_analyzed(class_name=Client, _all=False, name=name, num=num)
        return s
    return False

@login_router.post("/login")
@limiter.limit(f"{limite}/minute")
async def _login(request: Request, info_user:UserInfo):
    username = info_user.name
    user = DB.get_user(class_name=Client, name=username, _all=False)
    password = hashpw(info_user.password)
    age = info_user.age
    contact = info_user.contact
    salt = info_user.salt
    salt = salt.encode() if isinstance(salt, str) else salt
    checksalt(salt)
    token = create_token({"username": username}, key=salt)
    sucess = False
    if user:
        vp = checkpw(password=info_user.password, hashed=password)
        if vp:
            USERS.set(username, salt, expire=TTL)
        
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Mot de passe incorrecte"
                )
        sucess = True

    else:
        print('Ajout de l\'user.')
        fernet = FernetManager(password=password, salt=salt)
        age = fernet.encrypt(age)
        contact = fernet.encrypt(contact)
        sucess = DB.add_user(name=username, password=password, age=age, contact=contact)
        if sucess:
            print('User ajouté avec succèss')
            USERS.set(username, salt, expire=TTL)
        
    return {
        "new_user": not user,
        "success": sucess,
        "salt": salt,
        "username":username,
        "access_token": token
        }

@login_router.get('/health')
@limiter.limit(f"{limite}/minute")
async def _health(request: Request, username:str=Query(...), credentials:HTTPAuthorizationCredentials = Depends(baerer)):
    can = verify_username(username, credentials.credentials)
    if can:
        try:
            user_data = {
                }
            user = DB.get_user(class_name=Client, name=username, _all=False)
            if user and username in USERS:
                salt = USERS.get(username)
                fernet = FernetManager(password=user.password, salt=salt)
                user_data = {
                    'username': user.name,
                    'contact': fernet.decrypt(user.contact) if user.contact else "",
                    'age': fernet.decrypt(user.age) if user.age else "",
                    'num_analyzed': fernet.decrypt(user.num_analyzed) if user.num_analyzed and isinstance(user.num_analyzed, bytes) else "0"
                }
            else:
                user_data = {}
            
            return {
                "info_user": user_data,
                "config_model": model_params,
                "port": PORT,
                "ip": IP,
                "limite": f"{limite}/minutes"
                }
        except cryptography.fernet.InvalidToken:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Salt invalide, veuillez renseignez votre salt d'origine !"
                )
