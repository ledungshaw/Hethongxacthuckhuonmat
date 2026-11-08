"""
NODE 2 - SECURITY GATE (TOI UU TOC DO + FPS)
- Fix nhận diện: Ép độ phân giải gốc cho Camera
- Hiển thị: Thu nhỏ khung hình cho vừa màn hình
"""
import cv2
import time
import sys
import os
import threading
from config import Config
from core.camera_handler import CameraHandler
from core.face_recognizer import FaceRecognizer, draw_text_unicode
from core.sync_manager import SyncManager
from core.anti_spoofing import AntiSpoofingONNX
from utils.logger import get_logger

logger = get_logger(__name__)


class Node2SecurityGate:
    def __init__(self):
        self.camera = CameraHandler()
        self.face_recognizer = FaceRecognizer()
        self.sync_manager = SyncManager()
        self.anti_spoofing = AntiSpoofingONNX()

        self.liveness_detector = None
        self.running = True
        self.frame_skip = 0

        self.cached_display = None
        self.cached_result = None

        self.gate_pass_cache = {}
        self.GATE_PASS_COOLDOWN = 5

        self.spoof_cache = {}
        self.liveness_history = {}
        self._last_user_id = None

        self.verified_users = {}
        self.VERIFIED_TTL = 60

        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0

        self.ai_start_time = time.time()
        self.ai_frame_count = 0
        self.ai_fps = 0

        print("\n" + "="*55)
        print("🚪 NODE 2 - SECURITY GATE")
        print("="*55)

        self._check_connection()
        self._sync_data()
        
        # ⭐ KÍCH HOẠT TỰ ĐỘNG ĐỒNG BỘ MỖI 5 GIÂY
        self.sync_manager.start_auto_sync(self.face_recognizer)

    def _check_connection(self):
        if self.sync_manager.test_connection():
            print("✅ Connected to Node 1")
        else:
            print("⚠️ Cannot connect!")

    def _sync_data(self):
        faces = self.sync_manager.sync_from_server(force=True)
        if faces:
            encodings = [f['encoding'] for f in faces]
            user_ids = [f['user_id'] for f in faces]
            names = [f['name'] for f in faces]
            self.face_recognizer.set_known_faces(encodings, user_ids, names)
            print(f"✅ Loaded {len(faces)} faces")

    def _update_fps(self):
        self.fps_frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_start_time = time.time()

    def _update_ai_fps(self):
        self.ai_frame_count += 1
        elapsed = time.time() - self.ai_start_time
        if elapsed >= 1.0:
            self.ai_fps = self.ai_frame_count / elapsed
            self.ai_frame_count = 0
            self.ai_start_time = time.time()

    def _should_call_gate_pass(self, user_id):
        now = time.time()
        last = self.gate_pass_cache.get(user_id, 0)
        if now - last > self.GATE_PASS_COOLDOWN:
            self.gate_pass_cache[user_id] = now
            return True
        return False

    def _get_liveness_cached(self, frame, face_location, user_id):
        now = time.time()
        if user_id in self.spoof_cache:
            cached_live, cached_score, cached_time = self.spoof_cache[user_id]
            if (now - cached_time) < Config.LIVENESS_CACHE_TTL:
                return cached_live, cached_score

        if self._last_user_id != user_id:
            if user_id in self.liveness_history:
                self.liveness_history[user_id] = []
            self._last_user_id = user_id

        is_live, score = self.anti_spoofing.check_liveness(frame, face_location)

        if user_id not in self.liveness_history:
            self.liveness_history[user_id] = []
        history = self.liveness_history[user_id]
        history.append(score)
        if len(history) > Config.LIVENESS_SMOOTHING_WINDOW:
            history.pop(0)

        sorted_history = sorted(history)
        median_score = sorted_history[len(sorted_history) // 2]
        is_live_smooth = median_score >= Config.LIVENESS_THRESHOLD

        print(f"🎭 Anti-Spoof: real={score:.4f} | median={median_score:.4f} → {'THẬT' if is_live_smooth else 'GIẢ'}")

        self.spoof_cache[user_id] = (is_live_smooth, median_score, now)
        return is_live_smooth, median_score

    def run_recognition(self):
        if self.face_recognizer.get_match_count() == 0:
            print("⚠️ No faces! Chay sync truoc.")
            return

        print("\n🔍 RECOGNITION STARTED - Press 'q' to quit")
        
        cv2.namedWindow('Node 2 - Security Gate', cv2.WINDOW_NORMAL)

        while self.running:
            frame = self.camera.capture_frame()
            if frame is None:
                time.sleep(0.001)
                continue

            self._update_fps()
            self.frame_skip += 1

            result = self.face_recognizer.recognize_1n(frame)

            if result.get('message') == 'Dang cho...':
                if self.cached_display is not None:
                    display = self.cached_display.copy()
                else:
                    display = frame.copy()
                
                display = cv2.resize(display, (800, 600))
                cv2.putText(display, f"FPS: {self.current_fps:.1f}", (10, display.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow('Node 2 - Security Gate', display)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            self._update_ai_fps()
            display = frame.copy()

            if result['success']:
                face_location = result['face_location']
                raw_location = result.get('raw_box', face_location)
                temp_user_id = result['user_id'] if result['user_id'] else 0
                is_live, liveness_score = self._get_liveness_cached(frame, raw_location, temp_user_id)

                display = self.anti_spoofing.draw_liveness_info(display, face_location, is_live, liveness_score)

                if result['is_match'] and result['user_id'] is not None:
                    user_id = result['user_id']
                    display = self.face_recognizer.draw_face_info(display, face_location, True, result['user_name'], result.get('confidence', result.get('similarity', 0)))

                    if is_live:
                        if self._should_call_gate_pass(user_id):
                            self.sync_manager.gate_pass_async(user_id)
                        display = draw_text_unicode(display, "✅ DA XAC THUC", (10, 30), font_size=20, color=(0, 255, 0))
                    else:
                        display = draw_text_unicode(display, "⚠️ ANH GIA", (10, 30), font_size=20, color=(0, 0, 255))
                else:
                    display = self.face_recognizer.draw_face_info(display, face_location, False, similarity=result.get('similarity', 0))

            display = draw_text_unicode(display, f"Faces: {self.face_recognizer.get_match_count()} | FPS: {self.current_fps:.1f}",
                                        (10, display.shape[0] - 10), font_size=14, color=(255, 255, 255))

            self.cached_display = display.copy()
            self.cached_result = result

            display_show = cv2.resize(display, (800, 600))
            cv2.imshow('Node 2 - Security Gate', display_show)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

    def show_menu(self):
        print("\n" + "="*55)
        print("1. 🔍 Start Recognition")
        print("2. 🔄 Sync from Node 1")
        print("3. 📊 Show database")
        print("4. ❌ Exit")
        print("="*55)

    def run(self):
        while self.running:
            self.show_menu()
            choice = input("\n📌 Select: ").strip()

            if choice == '1':
                self.run_recognition()
            elif choice == '2':
                faces = self.sync_manager.sync_from_server(force=True)
                if faces:
                    self.face_recognizer.set_known_faces([f['encoding'] for f in faces], [f['user_id'] for f in faces], [f['name'] for f in faces])
            elif choice == '3':
                for i, (uid, name) in enumerate(zip(self.face_recognizer.known_user_ids, self.face_recognizer.known_names)):
                    print(f"   {i+1}. ID: {uid} - {name}")
            elif choice == '4':
                self.running = False
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    node = Node2SecurityGate()
    node.run()