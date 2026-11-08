"""
Sync Manager - KHOP NODE 1
- GET /api/admin/users
- POST /api/user/<id>/gate-pass
- GET /api/current_time
- POST /api/face-log
"""
import requests
import json
import pickle
import os
import time
import threading
from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class SyncManager:
    def __init__(self, server_url=Config.SERVER_URL):
        self.server_url = server_url
        self.cache_path = Config.ENCODINGS_PATH
        self.cache_expire = 300
        self.last_sync = 0
        self.session = requests.Session()
        self._auto_sync_running = False
        logger.info(f"✅ SyncManager initialized: {server_url}")
    
    def test_connection(self):
        try:
            # ⭐ SỬA: Bỏ /api bị lặp
            response = self.session.get(f"{self.server_url}/current_time", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"❌ Test connection failed: {e}")
            return False
    
    def sync_from_server(self, force=False):
        if not force and self._is_cache_valid():
            logger.info("📦 Using cached encodings")
            return self._load_cache()
        
        logger.info("🔄 Syncing from Node 1...")
        try:
            # ⭐ SỬA: Bỏ /api bị lặp
            response = self.session.get(f"{self.server_url}/admin/users", timeout=30)
            if response.status_code != 200:
                logger.error(f"❌ Server error: {response.status_code}")
                return self._load_cache()
            
            users = response.json()
            known_faces = []
            for user in users:
                if user.get('face_encoding'):
                    try:
                        encoding = json.loads(user['face_encoding'])
                        known_faces.append({
                            'user_id': user['id'],
                            'name': user.get('full_name', f"User_{user['id']}"),
                            'encoding': encoding
                        })
                    except Exception as e:
                        logger.warning(f"⚠️ Cannot parse encoding for user {user['id']}: {e}")
            
            self._save_cache(known_faces)
            self.last_sync = time.time()
            logger.info(f"✅ Synced {len(known_faces)} faces from Node 1")
            
            if known_faces:
                print(f"✅ Synced {len(known_faces)} faces, vector dim = {len(known_faces[0]['encoding'])}")
            return known_faces
        except Exception as e:
            logger.error(f"❌ Sync failed: {e}")
            return self._load_cache()

    def start_auto_sync(self, face_recognizer, interval=5):
        """Tự động đồng bộ vector từ Node 1 định kỳ chạy ngầm"""
        if self._auto_sync_running:
            return
        self._auto_sync_running = True

        def _loop():
            while self._auto_sync_running:
                time.sleep(interval)
                try:
                    faces = self.sync_from_server(force=True)
                    if faces and face_recognizer is not None:
                        face_recognizer.set_known_faces(
                            [f['encoding'] for f in faces],
                            [f['user_id'] for f in faces],
                            [f['name'] for f in faces]
                        )
                except Exception as e:
                    logger.error(f"❌ Auto sync error: {e}")

        thread = threading.Thread(target=_loop, daemon=True)
        thread.start()
    
    def _is_cache_valid(self):
        if not os.path.exists(self.cache_path):
            return False
        if time.time() - self.last_sync > self.cache_expire:
            return False
        try:
            with open(self.cache_path, 'rb') as f:
                data = pickle.load(f)
                return len(data) > 0
        except:
            return False
    
    def _load_cache(self):
        try:
            with open(self.cache_path, 'rb') as f:
                data = pickle.load(f)
                logger.info(f"📦 Loaded {len(data)} faces from cache")
                return data
        except Exception as e:
            logger.warning(f"⚠️ Cannot load cache: {e}")
            return []
    
    def _save_cache(self, data):
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"💾 Cached {len(data)} faces")
        except Exception as e:
            logger.warning(f"⚠️ Cannot save cache: {e}")
    
    def gate_pass(self, user_id):
        try:
            # ⭐ SỬA: Bỏ /api bị lặp
            response = self.session.post(
                f"{self.server_url}/user/{user_id}/gate-pass",
                timeout=10
            )
            if response.status_code == 200:
                logger.info(f"✅ Gate pass confirmed for user {user_id}")
                print(f"✅ Gate pass OK cho user {user_id}")
                return response.json()
            else:
                logger.warning(f"⚠️ Gate pass failed: {response.status_code} - {response.text}")
                print(f"⚠️ Gate pass failed user {user_id}: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"❌ Gate pass error: {e}")
            return None
    
    def gate_pass_async(self, user_id):
        def _call():
            try:
                self.gate_pass(user_id)
            except Exception as e:
                logger.error(f"❌ Async gate pass error: {e}")
        
        thread = threading.Thread(target=_call, daemon=True)
        thread.start()

    # ⭐ SỬA: Bỏ /api bị lặp
    def save_face_log_async(self, user_id, user_name, similarity, is_live, status, image_path=''):
        def _call():
            try:
                url = f"{self.server_url}/face-log"
                data = {
                    'user_id': user_id,
                    'user_name': user_name,
                    'similarity': similarity,
                    'is_live': is_live,
                    'status': status,
                    'image_path': image_path
                }
                self.session.post(url, json=data, timeout=5)
                logger.info(f"📸 Sent face log: {user_name} - {status}")
            except Exception as e:
                logger.error(f"❌ Send face log error: {e}")
        
        thread = threading.Thread(target=_call, daemon=True)
        thread.start()