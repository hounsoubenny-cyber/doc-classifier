#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 15:50:10 2026

@author: hounsousamuel
"""

import os, sys
sys.path.insert(1, os.path.dirname(os.path.abspath(os.path.join(__file__, ".."))))
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from classification_doc_avec_interface_react.analyze_router import router
from login_router import login_router
from limiter import limiter
from uvicorn import Config, Server
from config import IP, PORT, limite
from slowapi.errors import RateLimitExceeded
import threading
import atexit
import aiohttp
import asyncio
import time
import bcrypt
from datetime import datetime
import nest_asyncio
nest_asyncio.apply()

server = None

app = FastAPI(
    version="1.0",
    docs_url='/api/docs',
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    on_startup=[lambda: print("API lancée !")],
    on_shutdown=[lambda: print("API fermée !")],
    )
app.include_router(router=router, prefix="/api")
app.include_router(router=login_router, prefix="/api")
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=[
        "*"
        # "http://localhost:3000", "http://localhost:3001"
        # "http://0.0.0.0:3000", "http://0.0.0.0:3001"
        # "http://127.0.0.1:3000", "http://127.0.0.1:3001",
        # "http://127.0.0.1:3000"
    ],
    allow_headers=['*'],
    allow_methods=['*'],
    )

REACT_URL = "/static"
BUILD_URL = "/build"

app.mount(BUILD_URL, StaticFiles(directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "FRONT_END_REACT", "build")), name="build")
app.mount(REACT_URL, StaticFiles(directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "FRONT_END_REACT", "build", "static")), name="static")


@app.exception_handler(RateLimitExceeded)
async def _handler(request: Request, exc:RateLimitExceeded):
    return {
        "status_code": 400,
        "message": "Trop de requêtes, veuillez patientez !"
        }

@app.get('/api/close')
def _close_api():
    global server
    if server is None:
        print('Serveur non lancé !')
        return {
            "message ": "Serveur non lancé !"
            }
    else:
        __close_api()
        print('Serveur fermé.')
        return {
            "message ": 'Serveur fermé.'
            }
@app.get("/api/test")
def _test():
    return {
        "message": "Test de l'api !"
        }

@app.get("/api/salt")
@limiter.limit(f"{limite}/minute")
def _get_salt(request: Request):
    return {
        "salt": bcrypt.gensalt().decode(),
        "datetime": datetime.utcnow()
        }

@app.get("/api/model_state")
@limiter.limit(f"{limite}/minute")
async def _get_state(request: Request):
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ipython.html"))

@app.get("/")
async def _home():
    try:
        INDEX_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FRONT_END_REACT", "build", "index.html")
        return FileResponse(INDEX_HTML)
    except:
        return {
            "message": "API Anti-Phishing - Interface React non disponible",
            "api_docs": "/api/docs",
            "endpoints": {
                
                },
            }
    

@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    """Capture toutes les routes pour React Router"""
    excluded_prefixes = ["api/", "docs", "redoc", "openapi.json"]
    
    print(full_path)
    if any(full_path.startswith(prefix) for prefix in excluded_prefixes):
        raise HTTPException(404, detail="Route non trouvée")
        
  
    # raise HTTPException(status_code=404, detail="Route non trouvée")
    
def start(app, host, port):
    global server
    conf = Config(app=app, workers=10, host=host, port=port, loop='uvloop', use_colors=True)
    server = Server(config=conf)
    th = threading.Thread(target=server.run, daemon=True)
    return th, server

def __close_api():
    global server
    server.should_exit = True

async def close_api(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            print('Statut : ', response.status)

def close_api_atexit(url):
    def _close():
        asyncio.run(close_api(url))
    atexit.register(_close)
    

if __name__ == "__main__":
    print(os.path.join(os.path.dirname(os.path.abspath(__file__)), "FRONT_END_REACT", "build", "index.html"))
    print(os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "FRONT_END_REACT", "build", "index.html")))
    sys.exit(1)
    th, server = start(app, IP, PORT)
    th.start()
    time.sleep(2)
    URL = f"http://{IP}:{PORT}/api/"
    async def test():
        async with aiohttp.ClientSession() as session:
            async with session.get(URL+"test") as response:
                return {
                    "status_code": response.status,
                    "text": await response.text(),
                    "json": await response.json(),
                    "headers": dict(response.headers)
                    }
    async def test_login():
        login_url = URL+"login"
        data = {
            "name": "samuelhounsou",
            "password":"Kimetsu no yaiba",
            "age": 17,
            "contact": "01 92 62 54 31",
            "salt": bcrypt.gensalt().decode()
            }
        async with aiohttp.ClientSession() as session:
            async with session.post(login_url, json=data) as response:
                return await response.json()
    # print(asyncio.run(test()))
    print(asyncio.run(test_login()))
    time.sleep(4)
    asyncio.run(close_api(URL+"close"))
            
            
    
    