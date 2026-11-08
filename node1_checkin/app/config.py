# app/config.py
import os

class Config:
    SECRET_KEY = 'your-secret-key-change-in-production'
    DEBUG = True
    
    # Server trung tâm - IP 192.168.137.13
    SERVER_URL = 'http://192.168.137.13:5000/api'
    
    # Camera
    CAMERA_INDEX = 0
    
    # Upload
    UPLOAD_FOLDER = 'uploads/temp'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
