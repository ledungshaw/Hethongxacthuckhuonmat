"""
Face Detector - MTCNN (CHUẨN HÓA TỌA ĐỘ + 5-POINT ALIGNMENT + LỌC BOX RÁC)
- Trả về (top, right, bottom, left) chuẩn xác
- Căn chỉnh 5 điểm mốc (ArcFace 112x112) đồng bộ 100% quy trình với Node 1
"""
import cv2
import numpy as np
import threading
import os
from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    logger.warning("facenet-pytorch not installed, dung Haar fallback")

# ⭐ Ma trận 5 điểm chuẩn ArcFace/InsightFace cho ảnh 112x112 (Đồng bộ 100% với Node 1)
ARCFACE_SRC = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


class FaceDetectorMTCNN:
    def __init__(self):
        self.mtcnn = None
        self.lock = threading.Lock()
        self.available = False

        if MTCNN_AVAILABLE:
            try:
                self.mtcnn = MTCNN(
                    image_size=112,
                    margin=20,
                    min_face_size=Config.MTCNN_MIN_FACE_SIZE,
                    thresholds=Config.MTCNN_THRESHOLDS,
                    factor=Config.MTCNN_FACTOR,
                    post_process=True,
                    keep_all=False,
                    device='cpu',
                    select_largest=True
                )
                self.available = True
                print("✅ Node 2 FaceDetector: MTCNN (5-Point Alignment) loaded")
            except Exception as e:
                print(f"⚠️ MTCNN load failed: {e}")
                self.available = False

        # Tìm file Haar Cascade
        xml_path = '/usr/lib/python3/dist-packages/cv2/data/haarcascade_frontalface_default.xml'
        if not os.path.exists(xml_path):
            for p in ['/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml']:
                if os.path.exists(p):
                    xml_path = p
                    break

        if os.path.exists(xml_path):
            self.haar = cv2.CascadeClassifier(xml_path)
        else:
            try:
                self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            except Exception:
                self.haar = None

        if not self.available:
            print("⚠️ Node 2 FaceDetector: Dung Haar fallback")

    def _align_face_5points(self, img_rgb, landmarks):
        """Xoay và căn chỉnh khuôn mặt về chuẩn 112x112 bằng 5 điểm mốc"""
        dst = np.array(landmarks, dtype=np.float32).reshape(5, 2)
        tform, _ = cv2.estimateAffinePartial2D(dst, ARCFACE_SRC, method=cv2.LMEDS)
        if tform is None:
            return None
        aligned = cv2.warpAffine(img_rgb, tform, (112, 112), borderValue=0.0)
        return aligned.astype(np.uint8)

    def detect_and_align(self, frame_bgr):
        if frame_bgr is None:
            return None, None

        if self.available:
            with self.lock:
                try:
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    
                    # ⭐ Lấy box, độ tin cậy và 5 điểm mốc (landmarks)
                    boxes, probs, landmarks = self.mtcnn.detect(rgb, landmarks=True)
                    if boxes is None or len(boxes) == 0:
                        return None, None
                    
                    # ⭐ Lọc bỏ các box không chắc chắn (probs < 0.92)
                    valid_indices = [i for i, p in enumerate(probs) if p is not None and p > 0.92]
                    if len(valid_indices) == 0:
                        return None, None
                    
                    # Lấy box lớn nhất trong các box hợp lệ
                    areas = [(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) for i in valid_indices]
                    best_idx = valid_indices[int(np.argmax(areas))]
                    x1, y1, x2, y2 = boxes[best_idx]
                    
                    # ⭐ Ưu tiên căn chỉnh 5 điểm mốc đồng bộ với Node 1
                    aligned = None
                    if landmarks is not None and landmarks[best_idx] is not None:
                        aligned = self._align_face_5points(rgb, landmarks[best_idx])
                    
                    # Fallback nếu không lấy được landmarks thì mở rộng box 20% như cũ
                    if aligned is None:
                        w = x2 - x1
                        h = y2 - y1
                        margin = 0.2
                        crop_x1 = max(0, int(x1 - w * margin))
                        crop_y1 = max(0, int(y1 - h * margin))
                        crop_x2 = min(rgb.shape[1], int(x2 + w * margin))
                        crop_y2 = min(rgb.shape[0], int(y2 + h * margin))
                        
                        face_crop = rgb[crop_y1:crop_y2, crop_x1:crop_x2]
                        if face_crop.size == 0:
                            return None, None
                        aligned = cv2.resize(face_crop, (112, 112)).astype(np.uint8)
                    
                    # ⭐ Trả về box CHUẨN XÁC: (top, right, bottom, left)
                    h_img, w_img = frame_bgr.shape[:2]
                    top = max(0, int(y1))
                    left = max(0, int(x1))
                    bottom = min(h_img, int(y2))
                    right = min(w_img, int(x2))
                    box = (top, right, bottom, left)
                    
                    return aligned.astype(np.uint8), box
                except Exception as e:
                    logger.error(f"MTCNN detect failed: {e}")
                    return self._detect_haar(frame_bgr)
        else:
            return self._detect_haar(frame_bgr)

    def _detect_haar(self, frame_bgr):
        if frame_bgr is None or self.haar is None:
            return None, None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.haar.detectMultiScale(
            gray,
            scaleFactor=Config.HAAR_SCALE_FACTOR,
            minNeighbors=Config.HAAR_MIN_NEIGHBORS,
            minSize=Config.MIN_FACE_SIZE
        )
        if len(faces) == 0:
            return None, None
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        box = (y, x + w, y + h, x)
        face_crop = frame_bgr[y:y + h, x:x + w]
        if face_crop.size == 0:
            return None, None
        resized = cv2.resize(face_crop, (112, 112))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb, box

    def detect(self, frame):
        aligned, box = self.detect_and_align(frame)
        if box is None:
            return None
        return [box]

    def get_detector_name(self):
        return "MTCNN (CHUAN HOA)" if self.available else "Haar Cascade"


FaceDetectorHaar = FaceDetectorMTCNN