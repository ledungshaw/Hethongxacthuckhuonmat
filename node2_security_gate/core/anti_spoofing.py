"""
Anti-Spoofing - Phien ban ENSEMBLE MULTI-SCALE CHUAN XAC CAO NHAT
- Kết hợp 2 model: minifasnetv2_80x80.onnx (scale 2.7) + minifasnetv1se_80x80.onnx (scale 4.0)
- Fallback tự động về minifasnetv2.onnx nếu thiếu file
- Giữ chuẩn dải pixel [0.0, 255.0] của Silent-Face-Anti-Spoofing
"""
import cv2
import numpy as np
import onnxruntime as ort
import os
from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class AntiSpoofingONNX:
    # ⭐ Class 1 là REAL, Class 0 và 2 là FAKE/SPOOF
    REAL_INDEX = 1

    def __init__(self):
        self.models = []
        self.model_loaded = False
        self.real_index = self.REAL_INDEX

        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Ưu tiên nạp bộ đôi model Multi-Scale 80x80 có độ chính xác cao nhất
            v2_path = getattr(Config, 'MINIFASNET_V2_PATH', 'models/minifasnetv2_80x80.onnx')
            v1se_path = getattr(Config, 'MINIFASNET_V1SE_PATH', 'models/minifasnetv1se_80x80.onnx')

            candidates = [
                (v2_path, 2.7),
                (v1se_path, 4.0)
            ]

            for path, scale in candidates:
                if os.path.exists(path):
                    sess = ort.InferenceSession(path, sess_options=opts, providers=['CPUExecutionProvider'])
                    in_shape = sess.get_inputs()[0].shape
                    out_shape = sess.get_outputs()[0].shape
                    h = int(in_shape[2]) if len(in_shape) == 4 and in_shape[2] else 80
                    w = int(in_shape[3]) if len(in_shape) == 4 and in_shape[3] else 80
                    self.models.append({
                        'path': path,
                        'session': sess,
                        'input_name': sess.get_inputs()[0].name,
                        'output_name': sess.get_outputs()[0].name,
                        'input_size': (w, h),
                        'scale': scale,
                        'output_dim': int(out_shape[-1]) if len(out_shape) >= 2 else 1
                    })
                    print(f"✅ Anti-Spoof loaded: {path} (input={w}x{h}, scale={scale}, out={out_shape})")

            # Fallback về Config.MINIFASNET_PATH nếu không có 2 file 80x80
            if not self.models and os.path.exists(Config.MINIFASNET_PATH):
                sess = ort.InferenceSession(Config.MINIFASNET_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
                in_shape = sess.get_inputs()[0].shape
                out_shape = sess.get_outputs()[0].shape
                h = int(in_shape[2]) if len(in_shape) == 4 and in_shape[2] else 128
                w = int(in_shape[3]) if len(in_shape) == 4 and in_shape[3] else 128
                self.models.append({
                    'path': Config.MINIFASNET_PATH,
                    'session': sess,
                    'input_name': sess.get_inputs()[0].name,
                    'output_name': sess.get_outputs()[0].name,
                    'input_size': (w, h),
                    'scale': 2.7,
                    'output_dim': int(out_shape[-1]) if len(out_shape) >= 2 else 1
                })
                print(f"✅ Anti-Spoof fallback loaded: {Config.MINIFASNET_PATH} (input={w}x{h})")

            self.model_loaded = len(self.models) > 0
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            print(f"❌ Model load failed: {e}")
            self.model_loaded = False

        self.last_result = None

    def preprocess(self, frame, bbox, scale=2.7, input_size=(80, 80)):
        x1, y1, x2, y2 = bbox
        box_w, box_h = x2 - x1, y2 - y1
        if box_w <= 0 or box_h <= 0:
            return None

        h_img, w_img = frame.shape[:2]
        scale = min((h_img - 1) / box_h, min((w_img - 1) / box_w, scale))

        new_w = box_w * scale
        new_h = box_h * scale
        cx = box_w / 2.0 + x1
        cy = box_h / 2.0 + y1

        crop_x1 = max(0, int(cx - new_w / 2.0))
        crop_y1 = max(0, int(cy - new_h / 2.0))
        crop_x2 = min(w_img - 1, int(cx + new_w / 2.0))
        crop_y2 = min(h_img - 1, int(cy + new_h / 2.0))

        cropped = frame[crop_y1:crop_y2 + 1, crop_x1:crop_x2 + 1]
        if cropped.size == 0:
            return None

        resized = cv2.resize(cropped, input_size, interpolation=cv2.INTER_LINEAR)
        # ⭐ Chuẩn Silent-Face MiniFASNet: Giữ nguyên dải [0.0, 255.0], không chia 255.0
        bgr = resized.astype(np.float32)
        tensor = np.transpose(bgr, (2, 0, 1))
        return np.expand_dims(tensor, axis=0).astype(np.float32)

    def check_liveness(self, frame, face_location):
        if not self.model_loaded or frame is None or face_location is None:
            return False, 0.0

        try:
            top, right, bottom, left = face_location
            bbox = (left, top, right, bottom)

            real_scores = []
            fake_scores = []

            for m in self.models:
                input_tensor = self.preprocess(frame, bbox, scale=m['scale'], input_size=m['input_size'])
                if input_tensor is None:
                    continue

                outputs = m['session'].run(
                    [m['output_name']],
                    {m['input_name']: input_tensor}
                )
                logits = outputs[0][0]

                if m['output_dim'] >= 2:
                    exp_logits = np.exp(logits - np.max(logits))
                    probs = exp_logits / exp_logits.sum()
                    r_score = float(probs[self.real_index])
                    f_score = float(1.0 - r_score)
                else:
                    r_score = float(1.0 / (1.0 + np.exp(-logits[0])))
                    f_score = 1.0 - r_score

                real_scores.append(r_score)
                fake_scores.append(f_score)

            if not real_scores:
                return False, 0.0

            real_score = float(np.mean(real_scores))
            fake_score = float(np.mean(fake_scores))

            threshold = Config.LIVENESS_THRESHOLD
            is_live = real_score >= threshold

            # ⭐ IN LOG ĐỂ DEBUG
            print(f"🎭 Anti-Spoof: real={real_score:.4f}, fake={fake_score:.4f} → "
                  f"{'THẬT' if is_live else 'GIẢ'}")

            self.last_result = (is_live, real_score)
            return is_live, real_score

        except Exception as e:
            logger.error(f"Liveness check failed: {e}")
            return False, 0.0

    def draw_liveness_info(self, frame, face_location, is_live, score):
        if face_location is None:
            return frame

        top, right, bottom, left = face_location

        if not self.model_loaded:
            color = (128, 128, 128)
            label = "MODEL LOI"
        else:
            color = (0, 220, 0) if is_live else (0, 0, 255)
            label = f"{'✅ NGUOI THAT' if is_live else '⚠️ ANH GIA'} - {score*100:.1f}%"

        try:
            from core.face_recognizer import draw_text_unicode
            frame = draw_text_unicode(frame, label, (left, bottom + 8), font_size=16, color=color)
        except:
            cv2.putText(frame, label, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame