#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 17:02:09 2026

@author: hounsousamuel
"""

import os, sys
from sqlmodel import SQLModel, Field, Column, JSON, select, create_engine, Session
from dotenv import load_dotenv

class Client(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: int = Field(primary_key=True, default=None)
    name: str
    password: str 
    age: int = None
    num_analyzed: str = ""
    contact: str = ""
    history: dict = Field(sa_column=Column(JSON), default={})
   
    def __init__(self, *args, **kwg):
        super().__init__(*args, **kwg)
        self.validate_name()
        self.validate_age()
        
    def validate_name(self):
        if not self.name:
            raise ValueError('Le champs doit pas être vide !')
    
    def validate_age(self):
        if not self.age:
            raise ValueError('L\'age doit pas être vide et invalide !')
    
class DBManager:
    def __init__(self):
        load_dotenv()
        self.sql_uri = os.getenv("sql_uri")
        if not self.sql_uri:
            raise ValueError("Lien sql manquant, veuillez l'ajouter dans un fichier.env sql_uri=votre_lien !")
        
        self.engine = create_engine(self.sql_uri)
        SQLModel.metadata.create_all(self.engine)
        
    def add_user(self, name, password, age, contact:str="", history:dict={}):
        user = Client(name=name, password=password, age=age, history=history, contact=contact, num_analyzed="0")
        with Session(self.engine) as session:
            try:
                session.add(user)
                session.commit()
                session.refresh(user)
                # print("User : ", user)
                return True
            except Exception as e:
                print('Erreur dans l\'ajout : ', str(e))
                session.rollback()
                return False
    
    def delete_user(self, class_name, name:str = "", _id:int=None, _all:bool=False):
        with  Session(self.engine) as session:
            try:
                # q = None
                if _id:
                    q = select(class_name).where(class_name.id == _id)
                elif name:
                    q = select(class_name).where(class_name.name == name)
                elif name and _id:
                    q = select(class_name).where(class_name.id == _id | class_name.name == name)
                else:
                    raise ValueError("name ou id requis")
                r = session.exec(q)
                r = r.all() if _all else r.first()
                if r:
                    if _all:
                        for user in r:
                            session.delete(user)
                            
                    else:
                        session.delete(r)
                    
                    session.commit()
                return True
            except Exception as e:
                print('Erreur dans la suppression : ', str(e))
                session.rollback()
                return False
    
    def get_user(self, class_name, name:str = "", _id:int=None, _all:bool=False):
        with Session(self.engine) as session:
            try:
                if _id:
                    q = select(class_name).where(class_name.id == _id)
                elif name:
                    q = select(class_name).where(class_name.name == name)
                elif name and _id:
                    q = select(class_name).where(class_name.id == _id | class_name.name == name)
                else:
                    raise ValueError("name ou id requis")
                r = session.exec(q)
                r = r.all() if _all else r.first()
                return r
            
            except Exception as e:
                print('Erreur dans get_user : ', str(e))
                session.rollback()
                return None
    
    def update_history(self, class_name, name:str="", _id:int=None, _all:bool=False, history:dict={}):
        with Session(self.engine) as session:  # ← UNE SEULE session
            try:
                if _id:
                    q = select(class_name).where(class_name.id == _id)
                elif name:
                    q = select(class_name).where(class_name.name == name)
                elif name and _id:
                    q = select(class_name).where((class_name.id == _id) | (class_name.name == name))
                else:
                    raise ValueError("name ou id requis")
                
                r = session.exec(q)
                users = r.all() if _all else r.first()
                
                if not users:
                    return False
                
                if _all:
                    for user in users:
                        user.history.update(history)
                        session.add(user)
                else:
                    users.history.update(history)
                    session.add(users)
                
                session.commit()
                return True
                
            except Exception as e:
                print("Erreur dans la mise à jour de l'historique : ", str(e))
                session.rollback()
                return False
    
    def update_num_analyzed(self, class_name, name:str="", _id:int=None, _all:bool=False, num:int=0):
         with Session(self.engine) as session:  
             try:
                 if _id:
                     q = select(class_name).where(class_name.id == _id)
                 elif name:
                     q = select(class_name).where(class_name.name == name)
                 elif name and _id:
                     q = select(class_name).where((class_name.id == _id) | (class_name.name == name))
                 else:
                     raise ValueError("name ou id requis")
                 
                 r = session.exec(q)
                 users = r.all() if _all else r.first()
                 
                 if not users:
                     return False
                 
                 if _all:
                     for user in users:
                         user.num_analyzed = num
                         session.add(user)
                 else:
                     users.num_analyzed = num
                     session.add(users)
                 
                 session.commit()  
                 return True
                 
             except Exception as e:
                 print("Erreur dans la mise à jour de num_analyzed : ", str(e))
                 session.rollback()
                 return False
                
    

if __name__ == "__main__":
    c = Client(name="Samuel", password="SAM", age=15, contact="0000".encode())