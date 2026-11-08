"""
Node 2 - Web Server (Flask) - Bản tối ưu chống trễ
- Giới hạn FPS stream để nhường CPU cho AI
- Luôn gửi frame mới nhất, bỏ qua frame cũ
- Khung nhận diện chính xác
"""
import cv2
import time
import threading
import numpy as np
from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
from datetime import datetime
from config import Config
from core.camera_handler import CameraHandler
from core.face_recognizer import FaceRecognizer, draw_text_unicode
from core.sync_manager import SyncManager
from core.anti_spoofing import AntiSpoofingONNX
from utils.logger import get_logger

logger = get_logger(__name__)
app = Flask(__name__)
CORS(app)

class State:
    def __init__(self):
        self.camera = None
        self.face_recognizer = None
        self.sync_manager = None
        self.anti_spoofing = None
        
        self.raw_cond = threading.Condition()
        self.raw_frame = None
        self.raw_seq = 0
        
        self.overlay_lock = threading.Lock()
        self.overlay = None
        
        self.jpeg_cond = threading.Condition()
        self.latest_jpeg = None
        self.jpeg_seq = 0
        
        self.latest_result = {'status': 'idle', 'message': 'Dang cho...', 'fps': 0, 'faces': 0}
        self.history = []
        self.last_history_time = 0
        self.running = False
        self.threads = []
        self.spoof_cache = {}
        
        self.gate_pass_cache = {}
        self.GATE_PASS_COOLDOWN = 5
        
        self.last_log_time_per_user = {}
        self.LOG_COOLDOWN = 1
        
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0

state = State()

def sync_loop():
    while state.running:
        try:
            faces = state.sync_manager.sync_from_server(force=True)
            if faces:
                state.face_recognizer.set_known_faces([f['encoding'] for f in faces], [f['user_id'] for f in faces], [f['name'] for f in faces])
        except Exception:
            pass
        time.sleep(5)

def capture_loop():
    """Luồng chụp ảnh: Chụp liên tục với tốc độ cao nhất có thể"""
    while state.running:
        frame = state.camera.capture_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        
        state.fps_frame_count += 1
        elapsed = time.time() - state.fps_start_time
        if elapsed >= 1.0:
            state.current_fps = state.fps_frame_count / elapsed
            state.fps_frame_count = 0
            state.fps_start_time = time.time()
            
        with state.raw_cond:
            state.raw_frame = frame
            state.raw_seq += 1
            state.raw_cond.notify_all()

def ai_loop():
    """Luồng AI: Lấy frame mới nhất, xử lý và vẽ khung"""
    last_seq = 0
    while state.running:
        with state.raw_cond:
            state.raw_cond.wait_for(lambda: (not state.running) or (state.raw_seq > last_seq), timeout=1.0)
            if not state.running: break
            if state.raw_seq <= last_seq: continue
            frame = state.raw_frame
            last_seq = state.raw_seq

        try:
            # ⭐ Xử lý AI liên tục (không bỏ frame) để bắt khung nhanh nhất
            result = state.face_recognizer.recognize_1n(frame)
            canvas = np.zeros_like(frame)
            drawn = False
            status_data = {'status': 'idle', 'user_id': None, 'user_name': None, 'similarity': 0, 'is_live': False, 'message': 'Dang cho...'}
            
            if result['success']:
                face_loc = result['face_location']
                raw_loc = result.get('raw_box', face_loc)
                temp_user_id = result['user_id'] if result['user_id'] else 0
                now = time.time()
                
                if temp_user_id in state.spoof_cache:
                    is_live, l_score, c_time = state.spoof_cache[temp_user_id]
                    if now - c_time > 0.5:
                        is_live, l_score = state.anti_spoofing.check_liveness(frame, raw_loc)
                        state.spoof_cache[temp_user_id] = (is_live, l_score, now)
                else:
                    is_live, l_score = state.anti_spoofing.check_liveness(frame, raw_loc)
                    state.spoof_cache[temp_user_id] = (is_live, l_score, now)
                
                status_data['is_live'] = is_live
                color = (0, 0, 255)
                label = "Unknown"
                
                # ⭐ LUÔN LẤY TỌA ĐỘ CHUẨN
                top, right, bottom, left = face_loc

                if not is_live:
                    status_data.update({'status': 'spoof_detected', 'message': 'PHAT HIEN ANH GIA - TU CHOI', 'user_name': result['user_name'] if result['is_match'] else "Người lạ", 'similarity': result['similarity'] if result['is_match'] else 0})
                    label = "PHAT HIEN ANH GIA - TU CHOI"
                    color = (0, 0, 255)
                    
                    # ⭐ VẼ KHUNG ĐỎ ĐÚNG TỌA ĐỘ
                    cv2.rectangle(canvas, (left, top), (right, bottom), color, 3)
                    cv2.putText(canvas, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    drawn = True

                    log_user_name = result['user_name'] if result['is_match'] else "Người lạ"
                    log_user_id = result['user_id'] if result['is_match'] else 0
                    last_log = state.last_log_time_per_user.get(log_user_id, 0)
                    if now - last_log > state.LOG_COOLDOWN:
                        state.last_log_time_per_user[log_user_id] = now
                        log_similarity = result['similarity'] if result['is_match'] else 0.0
                        state.history.insert(0, {'time': datetime.now().strftime('%H:%M:%S'), 'user_name': log_user_name, 'is_live': False, 'similarity': log_similarity})
                        state.history = state.history[:10]
                        state.last_history_time = now
                        try:
                            state.sync_manager.save_face_log_async(user_id=log_user_id if log_user_id != 0 else None, user_name=log_user_name, similarity=log_similarity, is_live=False, status='spoof_detected')
                        except Exception: pass

                elif result['is_match']:
                    status_data.update({'status': 'confirmed', 'user_id': result['user_id'], 'user_name': result['user_name'], 'similarity': result['similarity'], 'message': 'XAC THUC THANH CONG'})
                    label = f"{result['user_name']} ({result['similarity']:.2f})"
                    color = (0, 255, 0)
                    
                    user_id = result['user_id']
                    if user_id:
                        last_call = state.gate_pass_cache.get(user_id, 0)
                        if now - last_call > state.GATE_PASS_COOLDOWN:
                            state.gate_pass_cache[user_id] = now
                            print(f"📡 [NODE 2] Gửi API Gate Pass cho user {user_id} sang Node 1...")
                            state.sync_manager.gate_pass_async(user_id)
                    
                    last_log = state.last_log_time_per_user.get(user_id, 0)
                    if now - last_log > state.LOG_COOLDOWN:
                        state.last_log_time_per_user[user_id] = now
                        state.history.insert(0, {'time': datetime.now().strftime('%H:%M:%S'), 'user_name': result['user_name'], 'is_live': True, 'similarity': result['similarity']})
                        state.history = state.history[:10]
                        state.last_history_time = now
                        try:
                            state.sync_manager.save_face_log_async(user_id=result['user_id'], user_name=result['user_name'], similarity=result['similarity'], is_live=True, status='confirmed')
                        except Exception: pass
                    
                    # ⭐ VẼ KHUNG XANH ĐÚNG TỌA ĐỘ
                    cv2.rectangle(canvas, (left, top), (right, bottom), color, 3)
                    cv2.putText(canvas, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    drawn = True
                else:
                    status_data.update({'status': 'unknown', 'similarity': result['similarity'], 'message': 'KHONG XAC DINH'})
                    label = f"Unknown ({result['similarity']:.2f})"
                    color = (0, 165, 255)
                    
                    last_log = state.last_log_time_per_user.get(0, 0)
                    if now - last_log > state.LOG_COOLDOWN:
                        state.last_log_time_per_user[0] = now
                        state.history.insert(0, {'time': datetime.now().strftime('%H:%M:%S'), 'user_name': "Người lạ", 'is_live': True, 'similarity': result['similarity']})
                        state.history = state.history[:10]
                        state.last_history_time = now
                        try:
                            state.sync_manager.save_face_log_async(user_id=None, user_name="Người lạ", similarity=result['similarity'], is_live=True, status='unknown')
                        except Exception: pass
                    
                    # ⭐ VẼ KHUNG CAM ĐÚNG TỌA ĐỘ
                    cv2.rectangle(canvas, (left, top), (right, bottom), color, 3)
                    cv2.putText(canvas, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    drawn = True
                
                status_data['fps'] = round(state.current_fps, 1)
                status_data['faces'] = state.face_recognizer.get_match_count()
                state.latest_result = status_data
            else:
                state.latest_result = status_data
                state.latest_result['fps'] = round(state.current_fps, 1)
                state.latest_result['faces'] = 0
            
            if drawn:
                mask = (canvas.max(axis=2) > 0).astype(np.uint8) * 255
                with state.overlay_lock:
                    state.overlay = {'canvas': canvas, 'mask': mask, 'time': time.time()}
            else:
                with state.overlay_lock:
                    state.overlay = None
        except Exception as e:
            logger.error(f"AI error: {e}")
        time.sleep(0.01)

def stream_loop():
    """Luồng Stream: Lấy frame gốc, vẽ overlay (nếu có) và nén thành JPEG"""
    last_seq = 0
    while state.running:
        with state.raw_cond:
            state.raw_cond.wait_for(lambda: (not state.running) or (state.raw_seq > last_seq), timeout=1.0)
            if not state.running: break
            if state.raw_seq <= last_seq: continue
            frame = state.raw_frame
            last_seq = state.raw_seq

        display = frame.copy()
        with state.overlay_lock:
            ov = state.overlay
        
        if ov is not None and (time.time() - ov['time']) < 1.5:
            cv2.copyTo(ov['canvas'], ov['mask'], display)

        display_small = cv2.resize(display, (640, 480))
        
        ret, jpeg = cv2.imencode('.jpg', display_small, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if ret:
            with state.jpeg_cond:
                state.latest_jpeg = jpeg.tobytes()
                state.jpeg_seq += 1
                state.jpeg_cond.notify_all()
        
        time.sleep(0.06)

def generate_mjpeg():
    last_seq = -1
    while state.running:
        with state.jpeg_cond:
            state.jpeg_cond.wait_for(lambda: (not state.running) or (state.jpeg_seq != last_seq), timeout=2.0)
            if state.jpeg_seq == last_seq or state.latest_jpeg is None: continue
            data = state.latest_jpeg
            last_seq = state.jpeg_seq
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/current_result')
def api_current_result():
    return jsonify(state.latest_result)

@app.route('/api/history')
def api_history():
    return jsonify(state.history)

def init_node():
    print("Khoi tao Node 2...")
    state.camera = CameraHandler()
    state.face_recognizer = FaceRecognizer()
    state.sync_manager = SyncManager()
    state.anti_spoofing = AntiSpoofingONNX()
    
    faces = state.sync_manager.sync_from_server(force=True)
    if faces:
        state.face_recognizer.set_known_faces([f['encoding'] for f in faces], [f['user_id'] for f in faces], [f['name'] for f in faces])

if __name__ == "__main__":
    init_node()
    state.running = True
    state.threads = [
        threading.Thread(target=sync_loop, daemon=True),
        threading.Thread(target=capture_loop, daemon=True),
        threading.Thread(target=ai_loop, daemon=True),
        threading.Thread(target=stream_loop, daemon=True)
    ]
    for t in state.threads: t.start()
    print("🌐 WEB SERVER ĐANG CHẠY TẠI: http://<IP_Pi>:5001")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True, use_reloader=False)