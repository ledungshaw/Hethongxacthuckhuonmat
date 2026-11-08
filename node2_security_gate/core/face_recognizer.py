"""
Face Recognizer - Tối ưu tốc độ cho Pi 4
- Bắt khung nhanh và chính xác
- Đồng bộ 100% ảnh aligned 5 điểm từ FaceDetector với Node 1
"""
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from config import Config
from utils.logger import get_logger
from core.face_detector import FaceDetectorMTCNN

logger = get_logger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

_FONT_CACHE = {}

def _get_font(size):
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.truetype(Config.FONT_PATH, size)
        except:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]

def draw_text_unicode(frame, text, position, font_size=22, color=(0, 255, 0)):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = _get_font(font_size)
        draw.text(position, text, font=font, fill=color)
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return frame
    except Exception as e:
        logger.error(f"❌ Draw text failed: {e}")
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

class FaceRecognizer:
    def __init__(self):
        self.threshold_high = Config.THRESHOLD
        self.threshold_low = Config.THRESHOLD_LOW
        self.known_encodings = []
        self.known_user_ids = []
        self.known_names = []
        self.known_encodings_np = None
        self.is_ready = False
        self.frame_count = 0
        self.last_result = None
        self.smoothed_box = None
        self.last_detect_frame = 0
        self.face_lost_ttl = Config.FACE_LOST_TTL
        self.smooth_factor = Config.SMOOTH_FACTOR
        self.faiss_index = None
        self.use_faiss = FAISS_AVAILABLE

        self.face_detector = FaceDetectorMTCNN()

        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = Config.ONNX_INTRA_THREADS
            opts.inter_op_num_threads = Config.ONNX_INTER_THREADS
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                Config.MOBILEFACENET_PATH,
                sess_options=opts,
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        except Exception as e:
            logger.error(f"❌ MobileFaceNet load failed: {e}")
            self.session = None

    def set_known_faces(self, encodings, user_ids, names):
        self.known_encodings = encodings
        self.known_user_ids = user_ids
        self.known_names = names
        self.is_ready = len(encodings) > 0

        if self.is_ready:
            enc_array = np.array(encodings, dtype=np.float32)
            norms = np.linalg.norm(enc_array, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            enc_array = enc_array / norms
            self.known_encodings_np = enc_array

        if self.use_faiss and len(encodings) > 0:
            try:
                dim = len(encodings[0])
                self.faiss_index = faiss.IndexFlatIP(dim)
                self.faiss_index.add(self.known_encodings_np)
            except Exception:
                self.use_faiss = False
                self.faiss_index = None

    def _smooth_box(self, new_box):
        if self.smoothed_box is None:
            self.smoothed_box = new_box
            return new_box
        top, right, bottom, left = new_box
        s_top, s_right, s_bottom, s_left = self.smoothed_box
        alpha = self.smooth_factor
        n_top = int(alpha * s_top + (1 - alpha) * top)
        n_right = int(alpha * s_right + (1 - alpha) * right)
        n_bottom = int(alpha * s_bottom + (1 - alpha) * bottom)
        n_left = int(alpha * s_left + (1 - alpha) * left)
        self.smoothed_box = (n_top, n_right, n_bottom, n_left)
        return self.smoothed_box

    def preprocess(self, face_bgr):
        face_img = cv2.resize(face_bgr, (112, 112))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img.astype(np.float32)
        face_img = (face_img - 127.5) / 127.5
        face_img = np.transpose(face_img, (2, 0, 1))
        face_img = np.expand_dims(face_img, axis=0)
        return face_img.astype(np.float32)

    def extract_face_vector(self, face_bgr):
        if self.session is None or face_bgr is None:
            return None
        try:
            input_tensor = self.preprocess(face_bgr)
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            embedding = outputs[0][0].copy()
            norm = np.linalg.norm(embedding)
            if norm < 1e-8:
                return None
            return embedding / norm
        except Exception:
            return None

    def recognize_1n(self, frame):
        if not self.is_ready or frame is None:
            return {'success': False, 'message': 'Chua san sang', 'face_location': None}

        self.frame_count += 1

        try:
            # ⭐ Dùng frame gốc để lấy ảnh aligned_rgb (đã xoay 5 điểm chuẩn) và raw_box
            aligned_rgb, raw_box = self.face_detector.detect_and_align(frame)

            if raw_box is None:
                if self.smoothed_box is not None:
                    if (self.frame_count - self.last_detect_frame) < self.face_lost_ttl:
                        face_location = self.smoothed_box
                    else:
                        self.smoothed_box = None
                        self.last_result = {'success': False, 'message': 'Khong tim thay', 'face_location': None}
                        return self.last_result
                else:
                    self.last_result = {'success': False, 'message': 'Khong tim thay', 'face_location': None}
                    return self.last_result
            else:
                face_location = self._smooth_box(raw_box)
                self.last_detect_frame = self.frame_count

            # ⭐ ĐÃ SỬA: Dùng trực tiếp ảnh aligned_rgb từ MTCNN (đồng bộ 100% với Node 1)
            # Chỉ fallback về cắt từ frame khi đang dùng box lưu tạm (smoothed_box)
            if aligned_rgb is not None:
                face_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)
            else:
                top, right, bottom, left = face_location
                face_bgr = frame[top:bottom, left:right]
                if face_bgr.size == 0:
                    return {'success': False, 'message': 'Vung mat trong', 'face_location': face_location}

            face_vector = self.extract_face_vector(face_bgr)
            if face_vector is None:
                return {'success': False, 'message': 'Loi vector', 'face_location': face_location}

            # So khớp
            if self.use_faiss and self.faiss_index is not None:
                query = face_vector.astype('float32').reshape(1, -1)
                distances, indices = self.faiss_index.search(query, k=1)
                best_idx = indices[0][0]
                sim = float(distances[0][0])
                all_matches = [{'user_id': self.known_user_ids[best_idx], 'user_name': self.known_names[best_idx], 'similarity': sim}]
            else:
                similarities = np.dot(self.known_encodings_np, face_vector)
                best_idx = int(np.argmax(similarities))
                sim = float(similarities[best_idx])
                all_matches = [{'user_id': self.known_user_ids[best_idx], 'user_name': self.known_names[best_idx], 'similarity': sim}]

            best_match = all_matches[0]
            is_match = best_match['similarity'] >= self.threshold_high

            result = {
                'success': True,
                'user_id': best_match['user_id'] if is_match else None,
                'user_name': best_match['user_name'] if is_match else None,
                'face_location': face_location,
                'raw_box': raw_box if raw_box is not None else face_location,
                'similarity': round(best_match['similarity'], 4),
                'confidence': round(best_match['similarity'], 4),
                'is_match': is_match,
                'message': '✅ KHỚP' if is_match else '❌ KHÔNG KHỚP'
            }
            self.last_result = result
            return result
        except Exception as e:
            logger.error(f"❌ Recognize failed: {e}")
            return {'success': False, 'message': str(e), 'face_location': None}

    def draw_face_info(self, frame, face_location, is_match=False, user_name=None, confidence=0, similarity=0):
        if face_location is None or frame is None:
            return frame
        top, right, bottom, left = face_location
        color = (0, 255, 0) if is_match else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
        if is_match and user_name:
            label = f"✅ {user_name} ({int(confidence*100)}%)"
        else:
            label = f"❌ KHÔNG XÁC ĐỊNH (sim={similarity:.3f})"
        frame = draw_text_unicode(frame, label, (left, top - 30), font_size=Config.FONT_SIZE, color=color)
        return frame

    def get_match_count(self):
        return len(self.known_encodings)