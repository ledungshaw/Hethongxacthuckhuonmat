from flask import Flask, request, jsonify, g, send_file
from flask_cors import CORS
from datetime import datetime, timedelta
import json
import sqlite3
import os
import base64
from io import BytesIO
from PIL import Image
import qrcode
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import hashlib
import random
import string
import cv2
import numpy as np
import onnxruntime as ort
from facenet_pytorch import MTCNN
import threading
import time

# ⭐ THÊM: Thư viện PDF417
try:
    import pdf417gen
    PDF417_AVAILABLE = True
except ImportError:
    PDF417_AVAILABLE = False
    print("⚠️ pdf417gen chưa cài. Chạy: pip install pdf417gen")

app = Flask(__name__)
CORS(app)

DB_PATH = 'database/airport.db'
os.makedirs('database', exist_ok=True)
os.makedirs('uploads/faces', exist_ok=True)
os.makedirs('uploads/qrcodes', exist_ok=True)   # Giữ nguyên tên cũ để tương thích
os.makedirs('uploads/temp', exist_ok=True)
os.makedirs('uploads/crops', exist_ok=True)

# ==================== CẤU HÌNH EMAIL ====================
EMAIL_SENDER = 'airport.system04@gmail.com'
EMAIL_PASSWORD = 'ntql unxw mpyt bqkp'

# ==================== MOBILEFACENET ONNX ====================
MOBILEFACENET_MODEL_PATH = 'models/mobilefacenet_new.onnx'

MOBILEFACENET_SESSION = None
MTCNN_DETECTOR = None
MTCNN_LOCK = threading.Lock()

# ⭐ Ma trận 5 điểm chuẩn ArcFace/InsightFace cho ảnh 112x112
ARCFACE_SRC = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def get_mobilefacenet_session():
    global MOBILEFACENET_SESSION
    if MOBILEFACENET_SESSION is None:
        try:
            if not os.path.exists(MOBILEFACENET_MODEL_PATH):
                print(f"❌ KHÔNG TÌM THẤY MODEL: {MOBILEFACENET_MODEL_PATH}")
                MOBILEFACENET_SESSION = None
                return None
            
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.inter_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            MOBILEFACENET_SESSION = ort.InferenceSession(
                MOBILEFACENET_MODEL_PATH,
                sess_options=opts,
                providers=['CPUExecutionProvider']
            )
            
            input_shape = MOBILEFACENET_SESSION.get_inputs()[0].shape
            output_shape = MOBILEFACENET_SESSION.get_outputs()[0].shape
            print(f"✅ MobileFaceNet ONNX loaded")
            print(f"   Input: {input_shape}")
            print(f"   Output: {output_shape}")
        except Exception as e:
            print(f"❌ MobileFaceNet load failed: {e}")
            MOBILEFACENET_SESSION = None
    return MOBILEFACENET_SESSION

def get_mtcnn_detector():
    global MTCNN_DETECTOR
    if MTCNN_DETECTOR is None:
        try:
            MTCNN_DETECTOR = MTCNN(
                image_size=112,
                margin=20,
                min_face_size=40,
                thresholds=[0.5, 0.6, 0.6],
                factor=0.709,
                post_process=True,
                keep_all=False,
                device='cpu',
                select_largest=True
            )
            print("✅ MTCNN detector loaded (size=112, post_process=True)")
        except Exception as e:
            print(f"⚠️ MTCNN load failed: {e}")
            MTCNN_DETECTOR = False
    return MTCNN_DETECTOR if MTCNN_DETECTOR is not False else None

# ==================== HÀM TẠO PNR ====================
def generate_pnr():
    chars = string.ascii_uppercase + string.digits
    chars = ''.join(c for c in chars if c not in 'O0I1')
    return ''.join(random.choice(chars) for _ in range(6))

def generate_seat_number():
    rows = random.randint(1, 30)
    cols = random.choice(['A', 'B', 'C', 'D', 'E', 'F'])
    return f"{rows}{cols}"

def generate_gate():
    gates = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10']
    return random.choice(gates)

# ==================== HÀM PREPROCESSING CHUẨN INSIGHTFACE 512-DIM ====================
def preprocess_face(face_img):
    """Preprocess cho MobileFaceNet 512-dim (InsightFace w600k_mbf)"""
    face_img = cv2.resize(face_img, (112, 112))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img.astype(np.float32)
    face_img = (face_img - 127.5) / 127.5
    face_img = np.transpose(face_img, (2, 0, 1))
    face_img = np.expand_dims(face_img, axis=0)
    return face_img.astype(np.float32)

# ==================== HÀM PHÁT HIỆN & CĂN CHỈNH MẶT (5-POINT ALIGNMENT) ====================
def detect_face_haar(img):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    x, y, w, h = faces[0]
    return img[y:y+h, x:x+w]

def align_face_5points(img_rgb, landmarks):
    """Xoay và căn chỉnh khuôn mặt về chuẩn 112x112 bằng 5 điểm mốc (2 mắt, mũi, 2 khóe miệng)"""
    dst = np.array(landmarks, dtype=np.float32).reshape(5, 2)
    tform, _ = cv2.estimateAffinePartial2D(dst, ARCFACE_SRC, method=cv2.LMEDS)
    if tform is None:
        return None
    aligned = cv2.warpAffine(img_rgb, tform, (112, 112), borderValue=0.0)
    return aligned.astype(np.uint8)

def detect_and_align_mtcnn(img):
    """Trả về ảnh RGB uint8 SẠCH (0-255) đã căn chỉnh 5 điểm chuẩn ArcFace 112x112"""
    mtcnn = get_mtcnn_detector()
    if mtcnn is None:
        return None
    
    with MTCNN_LOCK:
        try:
            # Reset MTCNN state để tránh cache frame cũ
            try:
                mtcnn.device = 'cpu'
            except:
                pass
            
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Bước 1: Detect box và 5 điểm landmarks
            boxes, probs, landmarks = mtcnn.detect(rgb, landmarks=True)
            if boxes is None or len(boxes) == 0:
                return None
            
            # Lọc các box có độ tin cậy >= 0.90
            valid_indices = [i for i, p in enumerate(probs) if p is not None and p >= 0.90]
            if not valid_indices:
                return None
            
            # Lấy box lớn nhất trong các box hợp lệ
            areas = [(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) for i in valid_indices]
            best_idx = valid_indices[int(np.argmax(areas))]
            
            # Bước 2: Ưu tiên căn chỉnh 5 điểm mốc (5-Point Affine Alignment)
            if landmarks is not None and landmarks[best_idx] is not None:
                aligned = align_face_5points(rgb, landmarks[best_idx])
                if aligned is not None:
                    return aligned
            
            # Bước 3: Fallback nếu không lấy được landmarks thì crop mở rộng
            x1, y1, x2, y2 = boxes[best_idx]
            w = x2 - x1
            h = y2 - y1
            margin = 0.2
            x1 = max(0, int(x1 - w * margin))
            y1 = max(0, int(y1 - h * margin))
            x2 = min(rgb.shape[1], int(x2 + w * margin))
            y2 = min(rgb.shape[0], int(y2 + h * margin))
            
            face_crop = rgb[y1:y2, x1:x2]
            if face_crop.size == 0:
                return None
            
            aligned = cv2.resize(face_crop, (112, 112))
            return aligned.astype(np.uint8)
        except Exception as e:
            print(f"⚠️ MTCNN align failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def check_image_quality(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, "Không đọc được ảnh"
        
        h, w = img.shape[:2]
        if min(h, w) < 50:
            return False, f"Ảnh quá nhỏ ({w}×{h})"
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness < 10 or brightness > 250:
            return False, f"Ảnh lỗi sáng (brightness={brightness:.1f})"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)

# ==================== TRÍCH XUẤT VECTOR ====================
def extract_face_vector_from_multiple(image_paths, user_id=None):
    """Trích xuất vector 512-dim từ nhiều ảnh"""
    try:
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        session = get_mobilefacenet_session()
        if session is None:
            print("❌ MobileFaceNet session không khả dụng")
            return None
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        all_encodings = []
        first_crop_saved = False
        print(f"🔍 Đang trích xuất MobileFaceNet từ {len(image_paths)} ảnh...")
        
        for idx, image_path in enumerate(image_paths):
            try:
                if not os.path.exists(image_path):
                    print(f"⚠️ Ảnh {idx+1} không tồn tại: {image_path}")
                    continue
                
                is_good, reason = check_image_quality(image_path)
                if not is_good:
                    print(f"⚠️ Ảnh {idx+1} cảnh báo: {reason}")
                    if "quá nhỏ" in reason or "không đọc" in reason:
                        continue
                
                img = cv2.imread(image_path)
                if img is None:
                    print(f"⚠️ Ảnh {idx+1} không đọc được")
                    continue
                print(f"   Ảnh {idx+1} shape: {img.shape}")
                
                face_rgb = detect_and_align_mtcnn(img)
                if face_rgb is None:
                    print(f"   ⚠️ MTCNN fail, dùng Haar")
                    face_bgr = detect_face_haar(img)
                    if face_bgr is not None:
                        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                
                # ⭐ ĐÃ SỬA: Nếu không tìm thấy mặt thì bỏ qua ảnh lỗi này, không lấy toàn bộ ảnh
                if face_rgb is None:
                    print(f"   ⚠️ Ảnh {idx+1}: Không tìm thấy khuôn mặt, bỏ qua ảnh này!")
                    continue
                
                if not first_crop_saved:
                    try:
                        crop_filename = f"crop_user_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        crop_full_path = os.path.join('uploads/crops', crop_filename)
                        save_crop = cv2.resize(face_rgb, (160, 160))
                        pil_img = Image.fromarray(save_crop)
                        pil_img.save(crop_full_path, 'JPEG', quality=92)
                        print(f"   ✅ Lưu crop (RGB qua PIL): {crop_full_path}")
                        first_crop_saved = True
                    except Exception as e:
                        print(f"   ⚠️ Lỗi lưu crop: {e}")
                
                face_bgr_for_preprocess = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
                face_input = preprocess_face(face_bgr_for_preprocess)
                
                outputs = session.run([output_name], {input_name: face_input})
                embedding = outputs[0][0]
                norm = np.linalg.norm(embedding)
                if norm < 1e-8:
                    continue
                embedding = embedding / norm
                all_encodings.append(embedding)
                print(f"   ✅ Ảnh {idx+1}: OK (vector length: {len(embedding)})")
            except Exception as e:
                print(f"   ❌ Ảnh {idx+1} lỗi: {e}")
                continue
        
        if not all_encodings:
            print("❌ Không có ảnh nào trích xuất được")
            return None
        
        print(f"📊 Tổng số vector trích xuất: {len(all_encodings)}")
        
        if len(all_encodings) >= 3:
            median_encoding = np.median(all_encodings, axis=0)
            median_encoding = median_encoding / np.linalg.norm(median_encoding)
            
            good_encodings = []
            for enc in all_encodings:
                sim = float(np.dot(enc, median_encoding))
                if sim >= 0.45:
                    good_encodings.append(enc)
            
            if len(good_encodings) >= 3:
                final_encoding = np.median(good_encodings, axis=0)
                final_encoding = final_encoding / np.linalg.norm(final_encoding)
                print(f"✅ Đã lấy MEDIAN của {len(good_encodings)}/{len(all_encodings)} vector (đã lọc outlier)")
            else:
                final_encoding = median_encoding
                print(f"✅ Đã lấy MEDIAN của {len(all_encodings)} vector (không đủ để lọc)")
        elif len(all_encodings) >= 1:
            avg_encoding = np.mean(all_encodings, axis=0)
            avg_encoding = avg_encoding / np.linalg.norm(avg_encoding)
            final_encoding = avg_encoding
            print(f"✅ Đã lấy MEAN của {len(all_encodings)} vector")
        else:
            return None
        
        return final_encoding.tolist()
    except Exception as e:
        print(f"❌ Lỗi trích xuất: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_face_vector_to_db(user_id, face_image_paths):
    try:
        if isinstance(face_image_paths, str):
            face_image_paths = [face_image_paths]
        vector = extract_face_vector_from_multiple(face_image_paths, user_id=user_id)
        if vector:
            conn = sqlite3.connect(DB_PATH, timeout=10)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            vector_json = json.dumps(vector)
            c.execute("UPDATE users SET face_encoding=? WHERE id=?", (vector_json, user_id))
            conn.commit()
            
            c.execute("SELECT LENGTH(face_encoding) FROM users WHERE id=?", (user_id,))
            row = c.fetchone()
            conn.close()
            
            if row and row[0] and row[0] > 0:
                print(f"✅ Đã lưu vector MobileFaceNet cho user {user_id} (JSON length={row[0]})")
            else:
                print(f"⚠️ Lưu vector thất bại cho user {user_id}")
            return vector
        return None
    except Exception as e:
        print(f"❌ Lỗi lưu vector face: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_face_images(face_images, user_id):
    saved_paths = []
    try:
        for idx, face_image in enumerate(face_images):
            img = Image.open(face_image).convert('RGB')
            filename = f"face_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}.jpg"
            filepath = os.path.join('uploads/faces', filename)
            img.save(filepath, 'JPEG', quality=92, optimize=True)
            saved_paths.append(filepath)
            print(f"✅ Đã lưu ảnh {idx+1}: {filepath}")
        return saved_paths
    except Exception as e:
        print(f"❌ Lỗi lưu ảnh: {e}")
        return []

# ==================== ⭐ HÀM SINH PDF417 (THAY THẾ QR) ====================
def generate_qr_code(user_id, flight_code, pnr):
    """
    ⭐ ĐÃ SỬA: Sinh mã PDF417 thay vì QR code.
    Tên hàm giữ nguyên (generate_qr_code) để không phá vỡ logic cũ.
    File output lưu vào uploads/qrcodes/ (giữ nguyên folder).
    """
    try:
        # Nội dung mã hóa (giống QR cũ)
        barcode_data = json.dumps({
            'user_id': user_id,
            'flight_code': flight_code,
            'pnr': pnr,
            'timestamp': datetime.now().isoformat()
        })
        
        # ⭐ Nếu pdf417gen không có, fallback về QR
        if not PDF417_AVAILABLE:
            print("⚠️ pdf417gen không có, fallback về QR")
            qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
            qr.add_data(barcode_data)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            filename = f"qr_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join('uploads/qrcodes', filename)
            img.save(filepath)
            print(f"✅ Đã tạo QR (fallback): {filepath}")
            return filepath
        
        # ⭐ Sinh PDF417
        codes = pdf417gen.encode(
            barcode_data,
            columns=6,          # Số cột (6-12, càng nhỏ càng cao)
            security_level=5    # Mức sửa lỗi (0-8, càng cao càng dày)
        )
        
        # Render thành ảnh PIL
        image = pdf417gen.render_image(
            codes,
            scale=3,            # Tỉ lệ phóng to
            ratio=3,            # Tỉ lệ chiều rộng/cao của từng module
            padding=15          # Viền trắng xung quanh
        )
        
        # Lưu file (giữ tên qr_ để tương thích với code cũ)
        filename = f"qr_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join('uploads/qrcodes', filename)
        image.save(filepath, 'PNG')
        print(f"✅ Đã tạo PDF417: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ Lỗi tạo PDF417: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==================== HÀM GỬI EMAIL ====================
def send_checkin_email(to_email, passenger_name, flight_code, flight_date, flight_time, pnr, seat, gate, qr_path):
    def _send():
        try:
            if not to_email or '@' not in to_email:
                print(f"❌ Email không hợp lệ: {to_email}")
                return False
            print(f"📧 Đang gửi email đến: {to_email}")
            
            msg = MIMEMultipart('related')
            msg['From'] = f'✈️ Hệ thống sân bay <{EMAIL_SENDER}>'
            msg['To'] = to_email
            msg['Subject'] = f'✈️ XÁC NHẬN CHECK-IN - Chuyến bay {flight_code} - PNR: {pnr}'
            
            try:
                formatted_date = datetime.strptime(flight_date, '%Y-%m-%d').strftime('%d/%m/%Y') if flight_date else datetime.now().strftime('%d/%m/%Y')
            except:
                formatted_date = flight_date if flight_date else datetime.now().strftime('%d/%m/%Y')
            formatted_time = flight_time if flight_time else datetime.now().strftime('%H:%M')
            now = datetime.now()
            
            # ⭐ ĐÃ SỬA: HTML email giữ nguyên cấu trúc, chỉ đổi chữ "Mã QR" thành "Mã PDF417"
            html_body = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Xác nhận Check-in</title>
</head>
<body style="margin:0;padding:0;background-color:#eef1f8;font-family:-apple-system,'Segoe UI',Arial,Tahoma,sans-serif;-webkit-font-smoothing:antialiased;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background-color:#eef1f8;padding:20px 10px;">
  <tr>
    <td align="center">
      <table role="presentation" width="600" cellpadding="0" cellspacing="0" border="0" style="max-width:600px;width:100%;background:#ffffff;border-radius:18px;overflow:hidden;box-shadow:0 8px 30px rgba(102,126,234,0.15);">
        
        <tr>
          <td align="center" style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);background-color:#667eea;padding:40px 25px;color:#ffffff;">
            <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%">
              <tr>
                <td align="center">
                  <div style="font-size:44px;line-height:1;margin-bottom:12px;">✈️</div>
                  <div style="font-size:26px;font-weight:800;color:#ffffff;letter-spacing:0.5px;margin:0 0 10px 0;text-transform:uppercase;">XÁC NHẬN CHECK-IN</div>
                  <div style="font-size:15px;color:#ffffff;opacity:0.95;">Chuyến bay: <strong>{flight_code}</strong></div>
                </td>
              </tr>
            </table>
          </td>
        </tr>
        
        <tr>
          <td style="padding:30px 25px 10px 25px;">
            <p style="margin:0 0 12px 0;font-size:16px;color:#333;line-height:1.6;">Kính gửi <strong style="color:#667eea;">{passenger_name}</strong>,</p>
            <p style="margin:0 0 25px 0;font-size:15px;color:#333;line-height:1.6;">Yêu cầu check-in của bạn đã được <strong style="color:#28a745;">DUYỆT</strong> thành công!</p>
            
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#f0f4ff;border:2px solid #667eea;border-radius:14px;margin:0 0 20px 0;">
              <tr>
                <td align="center" style="padding:25px 15px;">
                  <div style="font-size:12px;color:#667eea;font-weight:700;letter-spacing:2px;margin-bottom:10px;">🎫 THÔNG TIN LÊN MÁY BAY</div>
                  <div style="font-size:30px;font-weight:800;color:#667eea;letter-spacing:4px;">PNR: {pnr}</div>
                </td>
              </tr>
            </table>
            
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="margin:0 0 20px 0;border-collapse:separate;border-spacing:8px 0;">
              <tr>
                <td width="50%" align="center" valign="middle" style="background:#f8f9ff;border-radius:12px;padding:18px 10px;border:1px solid #e0e4f0;">
                  <div style="font-size:13px;color:#666;margin-bottom:8px;">💺 Số ghế</div>
                  <div style="font-size:26px;font-weight:800;color:#333;">{seat}</div>
                </td>
                <td width="50%" align="center" valign="middle" style="background:#f8f9ff;border-radius:12px;padding:18px 10px;border:1px solid #e0e4f0;">
                  <div style="font-size:13px;color:#666;margin-bottom:8px;">🚪 Cổng</div>
                  <div style="font-size:26px;font-weight:800;color:#333;">{gate}</div>
                </td>
              </tr>
            </table>
            
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#fafbff;border-left:4px solid #667eea;border-radius:10px;margin:0 0 20px 0;">
              <tr>
                <td style="padding:20px 22px;">
                  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
                    <tr>
                      <td style="padding:7px 0;font-size:14px;color:#666;width:150px;">✈️ Mã chuyến bay</td>
                      <td style="padding:7px 0;font-size:14px;color:#333;font-weight:700;">{flight_code}</td>
                    </tr>
                    <tr>
                      <td style="padding:7px 0;font-size:14px;color:#666;">📅 Ngày bay</td>
                      <td style="padding:7px 0;font-size:14px;color:#333;font-weight:700;">{formatted_date}</td>
                    </tr>
                    <tr>
                      <td style="padding:7px 0;font-size:14px;color:#666;">🕐 Giờ bay</td>
                      <td style="padding:7px 0;font-size:14px;color:#333;font-weight:700;">{formatted_time}</td>
                    </tr>
                    <tr>
                      <td style="padding:7px 0;font-size:14px;color:#666;">🎫 Mã PNR</td>
                      <td style="padding:7px 0;font-size:14px;color:#667eea;font-weight:800;">{pnr}</td>
                    </tr>
                    <tr>
                      <td style="padding:7px 0;font-size:14px;color:#666;">💺 Số ghế</td>
                      <td style="padding:7px 0;font-size:14px;color:#333;font-weight:700;">{seat}</td>
                    </tr>
                    <tr>
                      <td style="padding:7px 0;font-size:14px;color:#666;">🚪 Cổng</td>
                      <td style="padding:7px 0;font-size:14px;color:#333;font-weight:700;">{gate}</td>
                    </tr>
                  </table>
                </td>
              </tr>
            </table>
            
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#fafbff;border:2px dashed #c9d1e8;border-radius:14px;margin:0 0 20px 0;">
              <tr>
                <td align="center" style="padding:25px 20px;">
                  <div style="font-size:15px;font-weight:700;color:#333;margin-bottom:18px;">📱 Mã PDF417 lên máy bay</div>
                  <div style="display:inline-block;padding:8px;background:#ffffff;border:3px solid #667eea;border-radius:14px;">
                    <img src="cid:qr_code" alt="PDF417 Barcode" style="display:block;border-radius:8px;max-width:100%;height:auto;width:280px;" />
                  </div>
                  <div style="font-size:12px;color:#888;margin-top:18px;line-height:1.5;">Vui lòng xuất trình mã PDF417 này tại cổng lên máy bay</div>
                </td>
              </tr>
            </table>
            
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#fff8e1;border-left:4px solid #ffb300;border-radius:10px;margin:0 0 25px 0;">
              <tr>
                <td style="padding:16px 20px;font-size:13px;color:#7a5c00;line-height:1.6;">
                  ⚠️ <strong>Lưu ý:</strong> Vui lòng mang theo giấy tờ tùy thân và đến sân bay trước giờ bay ít nhất 45 phút.
                </td>
              </tr>
            </table>
            
            <p style="margin:0 0 10px 0;font-size:15px;color:#333;text-align:center;">Chúc bạn có một chuyến bay vui vẻ! 🎉</p>
          </td>
        </tr>
        
        <tr>
          <td align="center" style="padding:20px 25px;background:#f8f9ff;border-top:1px solid #eee;">
            <p style="margin:0;font-size:12px;color:#999;">© {now.year} Hệ thống sân bay - Airport Management System</p>
          </td>
        </tr>
        
      </table>
    </td>
  </tr>
</table>
</body>
</html>"""
            
            msg.attach(MIMEText(html_body, 'html', 'UTF-8'))
            
            if os.path.exists(qr_path):
                with open(qr_path, 'rb') as f:
                    img_data = f.read()
                    img = MIMEImage(img_data)
                    img.add_header('Content-ID', '<qr_code>')
                    img.add_header('Content-Disposition', 'inline', filename='boarding_pass.png')
                    msg.attach(img)
            
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            print(f"✅ Đã gửi email thành công đến {to_email}")
            return True
        except Exception as e:
            print(f"❌ Lỗi gửi email: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    thread = threading.Thread(target=_send)
    thread.daemon = True
    thread.start()
    return True

# ==================== DATABASE ====================
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        email TEXT,
        phone TEXT,
        full_name TEXT,
        cccd TEXT UNIQUE,
        role TEXT DEFAULT 'user',
        status TEXT DEFAULT 'pending',
        flight_code TEXT,
        flight_date TEXT,
        flight_time TEXT,
        ticket_type TEXT,
        luggage_count INTEGER DEFAULT 0,
        luggage_weight REAL DEFAULT 0,
        checked_in INTEGER DEFAULT 0,
        gate_passed INTEGER DEFAULT 0,
        boarded INTEGER DEFAULT 0,
        face_encoding TEXT,
        face_image_path TEXT,
        qr_code_path TEXT,
        qr_code_data TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS checkin_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        flight_code TEXT,
        flight_date TEXT,
        flight_time TEXT,
        ticket_type TEXT,
        luggage_count INTEGER DEFAULT 0,
        luggage_weight REAL DEFAULT 0,
        pnr_code TEXT,
        seat_number TEXT,
        gate_number TEXT,
        status TEXT DEFAULT 'pending',
        review_note TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        reviewed_at TIMESTAMP
    )''')
    
    c.execute("PRAGMA table_info(checkin_requests)")
    columns = [col[1] for col in c.fetchall()]
    
    if 'pnr_code' not in columns:
        c.execute("ALTER TABLE checkin_requests ADD COLUMN pnr_code TEXT")
    if 'seat_number' not in columns:
        c.execute("ALTER TABLE checkin_requests ADD COLUMN seat_number TEXT")
    if 'gate_number' not in columns:
        c.execute("ALTER TABLE checkin_requests ADD COLUMN gate_number TEXT")
    if 'origin' not in columns:
        c.execute("ALTER TABLE checkin_requests ADD COLUMN origin TEXT")
    if 'destination' not in columns:
        c.execute("ALTER TABLE checkin_requests ADD COLUMN destination TEXT")
    if 'fids_id' not in columns:
        c.execute("ALTER TABLE checkin_requests ADD COLUMN fids_id INTEGER")
    
    c.execute('''CREATE TABLE IF NOT EXISTS fids (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        flight_code TEXT NOT NULL,
        airline TEXT,
        origin TEXT,
        destination TEXT,
        flight_date TEXT,
        scheduled_time TEXT,
        estimated_time TEXT,
        actual_time TEXT,
        status TEXT DEFAULT 'Scheduled',
        gate TEXT,
        checkin_counter TEXT,
        terminal TEXT,
        remark TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    c.execute("PRAGMA table_info(fids)")
    fids_cols = [col[1] for col in c.fetchall()]
    if 'flight_date' not in fids_cols:
        c.execute("ALTER TABLE fids ADD COLUMN flight_date TEXT")
    
    c.execute('''CREATE TABLE IF NOT EXISTS fids_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fids_id INTEGER,
        old_status TEXT,
        new_status TEXT,
        changed_by TEXT,
        changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS activity_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        action TEXT,
        details TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    c.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_users_status ON users(status)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_checkin_user_id ON checkin_requests(user_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_checkin_status ON checkin_requests(status)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_activity_user_id ON activity_logs(user_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_fids_flight_code ON fids(flight_code)")
    
    from werkzeug.security import generate_password_hash
    c.execute("SELECT * FROM users WHERE username='admin'")
    if not c.fetchone():
        c.execute('''INSERT INTO users 
            (username, password, email, phone, full_name, cccd, role, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            ('admin', generate_password_hash('admin123'), 'admin@system.com', '0900000000', 'Administrator', '000000000000', 'admin', 'approved'))
    
    c.execute("SELECT COUNT(*) FROM fids")
    if c.fetchone()[0] == 0:
        today = datetime.now().strftime('%Y-%m-%d')
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        day_after = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
        
        sample_flights = [
            ('VN123', 'Vietnam Airlines', 'Hà Nội', 'TP.HCM', today, '08:00', '08:00', None, 'Scheduled', 'G1', 'C1', 'T1', None),
            ('VJ456', 'VietJet Air', 'TP.HCM', 'Đà Nẵng', today, '09:30', '09:30', None, 'Scheduled', 'G2', 'C2', 'T1', None),
            ('VN789', 'Vietnam Airlines', 'Hà Nội', 'Đà Nẵng', today, '10:45', '10:45', None, 'Boarding', 'G3', 'C3', 'T1', None),
            ('VJ012', 'VietJet Air', 'Đà Nẵng', 'TP.HCM', today, '12:00', '12:00', None, 'Scheduled', 'G4', 'C4', 'T1', None),
            ('VN345', 'Vietnam Airlines', 'Hà Nội', 'TP.HCM', today, '13:15', '13:15', None, 'Scheduled', 'G5', 'C5', 'T1', None),
            ('BAM678', 'Bamboo Airways', 'TP.HCM', 'Hà Nội', today, '14:30', '14:30', None, 'Delayed', 'G6', 'C6', 'T1', 'Delay 30 phút'),
            ('VN888', 'Vietnam Airlines', 'Hà Nội', 'TP.HCM', tomorrow, '08:00', '08:00', None, 'Scheduled', 'G1', 'C1', 'T1', None),
            ('VN999', 'Vietnam Airlines', 'Hà Nội', 'Phú Quốc', tomorrow, '15:30', '15:30', None, 'Scheduled', 'G2', 'C2', 'T1', None),
            ('VJ234', 'VietJet Air', 'TP.HCM', 'Hà Nội', day_after, '07:00', '07:00', None, 'Scheduled', 'G3', 'C3', 'T1', None),
            ('BAM111', 'Bamboo Airways', 'Đà Nẵng', 'Hà Nội', day_after, '11:30', '11:30', None, 'Scheduled', 'G4', 'C4', 'T1', None),
        ]
        for f in sample_flights:
            c.execute('''INSERT INTO fids 
                (flight_code, airline, origin, destination, flight_date, scheduled_time, estimated_time, actual_time, status, gate, checkin_counter, terminal, remark)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', f)
        print(f"✅ Đã seed {len(sample_flights)} chuyến bay")
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

init_db()

# ==================== FIDS API ====================
@app.route('/api/fids', methods=['GET'])
def get_fids():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM fids ORDER BY flight_date, scheduled_time")
    flights = c.fetchall()
    return jsonify([dict(f) for f in flights])

@app.route('/api/fids/<int:fids_id>', methods=['GET'])
def get_fids_detail(fids_id):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM fids WHERE id=?", (fids_id,))
    flight = c.fetchone()
    if not flight:
        return jsonify({'error': 'Không tìm thấy chuyến bay'}), 404
    return jsonify(dict(flight))

@app.route('/api/fids', methods=['POST'])
def create_fids():
    try:
        data = request.json
        conn = get_db()
        c = conn.cursor()
        c.execute('''INSERT INTO fids 
            (flight_code, airline, origin, destination, flight_date, scheduled_time, estimated_time, actual_time, status, gate, checkin_counter, terminal, remark)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (data['flight_code'], data.get('airline', ''), data.get('origin', ''),
             data.get('destination', ''), data.get('flight_date', ''),
             data.get('scheduled_time', ''),
             data.get('estimated_time', ''), data.get('actual_time', ''),
             data.get('status', 'Scheduled'), data.get('gate', ''),
             data.get('checkin_counter', ''), data.get('terminal', ''),
             data.get('remark', '')))
        conn.commit()
        return jsonify({'success': True, 'message': 'Đã thêm chuyến bay'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/fids/<int:fids_id>', methods=['PUT'])
def update_fids(fids_id):
    try:
        data = request.json
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT status FROM fids WHERE id=?", (fids_id,))
        old = c.fetchone()
        old_status = old['status'] if old else None
        c.execute('''UPDATE fids SET 
            flight_code=?, airline=?, origin=?, destination=?, flight_date=?,
            scheduled_time=?, estimated_time=?, actual_time=?, 
            status=?, gate=?, checkin_counter=?, terminal=?, remark=?,
            updated_at=CURRENT_TIMESTAMP
            WHERE id=?''',
            (data['flight_code'], data.get('airline', ''), data.get('origin', ''),
             data.get('destination', ''), data.get('flight_date', ''),
             data.get('scheduled_time', ''),
             data.get('estimated_time', ''), data.get('actual_time', ''),
             data.get('status', 'Scheduled'), data.get('gate', ''),
             data.get('checkin_counter', ''), data.get('terminal', ''),
             data.get('remark', ''), fids_id))
        if old_status and old_status != data.get('status'):
            c.execute('''INSERT INTO fids_logs (fids_id, old_status, new_status, changed_by)
                VALUES (?, ?, ?, ?)''',
                (fids_id, old_status, data.get('status'), data.get('changed_by', 'admin')))
        conn.commit()
        return jsonify({'success': True, 'message': 'Đã cập nhật chuyến bay'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/fids/<int:fids_id>', methods=['DELETE'])
def delete_fids(fids_id):
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("DELETE FROM fids WHERE id=?", (fids_id,))
        conn.commit()
        return jsonify({'success': True, 'message': 'Đã xóa chuyến bay'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== AUTH ====================
@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.json
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (data['username'],))
        if c.fetchone():
            return jsonify({'error': 'Tên đăng nhập đã tồn tại'}), 400
        from werkzeug.security import generate_password_hash
        c.execute('''INSERT INTO users 
            (username, password, email, phone, full_name, cccd, role, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (data['username'], generate_password_hash(data['password']), 
             data['email'], data['phone'], data['full_name'], data['cccd'], 'user', 'pending'))
        conn.commit()
        user_id = c.lastrowid
        c.execute("INSERT INTO activity_logs (user_id, action, details) VALUES (?, ?, ?)",
                  (user_id, 'register', 'Đăng ký tài khoản thành công'))
        conn.commit()
        return jsonify({
            'success': True,
            'message': 'Đăng ký thành công! Vui lòng chờ admin duyệt.',
            'user_id': user_id
        }), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.json
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (data['username'],))
        user = c.fetchone()
        if not user:
            return jsonify({'error': 'Tên đăng nhập không tồn tại'}), 401
        from werkzeug.security import check_password_hash
        if not check_password_hash(user['password'], data['password']):
            return jsonify({'error': 'Mật khẩu không đúng'}), 401
        if user['status'] == 'pending':
            return jsonify({'error': 'Tài khoản đang chờ admin duyệt'}), 403
        c.execute("INSERT INTO activity_logs (user_id, action, details) VALUES (?, ?, ?)",
                  (user['id'], 'login', 'Đăng nhập thành công'))
        conn.commit()
        return jsonify({
            'success': True,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'role': user['role'],
                'full_name': user['full_name'],
                'status': user['status']
            }
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== ADMIN ====================
@app.route('/api/admin/users', methods=['GET'])
def get_users():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM users ORDER BY id DESC")
    users = c.fetchall()
    result = []
    for u in users:
        user_dict = dict(u)
        if user_dict.get('face_image_path'):
            try:
                paths = json.loads(user_dict['face_image_path'])
                if isinstance(paths, list) and paths:
                    first_path = paths[0]
                else:
                    first_path = user_dict['face_image_path']
            except:
                first_path = user_dict['face_image_path']
            
            if os.path.exists(first_path):
                user_dict['face_image_url'] = f"/api/face_image/{user_dict['id']}"
            else:
                user_dict['face_image_url'] = None
        else:
            user_dict['face_image_url'] = None
        result.append(user_dict)
    return jsonify(result)

@app.route('/api/face_image/<int:user_id>')
def get_face_image(user_id):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT face_image_path FROM users WHERE id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    
    if not row or not row['face_image_path']:
        return '', 404
    
    path = row['face_image_path']
    try:
        paths = json.loads(path)
        if isinstance(paths, list) and paths:
            path = paths[0]
    except:
        pass
    
    try:
        return send_file(path, mimetype='image/jpeg')
    except:
        return '', 404

@app.route('/api/face_crop/<int:user_id>')
def get_face_crop(user_id):
    """Trả về ảnh crop khuôn mặt đã được trích xuất vector"""
    try:
        crop_dir = 'uploads/crops'
        if not os.path.exists(crop_dir):
            return '', 404
        
        pattern = f"crop_user_{user_id}_"
        matching = [f for f in os.listdir(crop_dir) if f.startswith(pattern)]
        if not matching:
            return '', 404
        
        matching.sort(reverse=True)
        crop_path = os.path.join(crop_dir, matching[0])
        
        if os.path.exists(crop_path):
            return send_file(crop_path, mimetype='image/jpeg')
        return '', 404
    except Exception as e:
        print(f"❌ Lỗi get_face_crop: {e}")
        return '', 404

@app.route('/api/admin/users/<int:user_id>/approve', methods=['POST'])
def approve_user(user_id):
    try:
        action = request.json.get('action')
        conn = get_db()
        c = conn.cursor()
        status = 'approved' if action == 'approve' else 'rejected'
        c.execute("UPDATE users SET status=? WHERE id=?", (status, user_id))
        conn.commit()
        c.execute("INSERT INTO activity_logs (user_id, action, details) VALUES (?, ?, ?)",
                  (user_id, f'admin_{action}', f'Admin {("duyệt" if action == "approve" else "từ chối")} tài khoản'))
        conn.commit()
        return jsonify({
            'success': True,
            'message': f'Đã {("duyệt" if action == "approve" else "từ chối")} tài khoản'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/stats', methods=['GET'])
def get_stats():
    conn = get_db()
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM users")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM users WHERE status='pending'")
    pending = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM users WHERE checked_in=1")
    checked_in = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM users WHERE gate_passed=1")
    gate_passed = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM users WHERE boarded=1")
    boarded = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM checkin_requests WHERE status='pending'")
    pending_checkin = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM checkin_requests WHERE status='approved'")
    approved_checkin = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM checkin_requests WHERE status='rejected'")
    rejected_checkin = c.fetchone()[0]
    
    c.execute("""
        SELECT flight_code, 
               COUNT(*) as total, 
               SUM(checked_in) as checked_in, 
               SUM(gate_passed) as gate_passed,
               SUM(boarded) as boarded 
        FROM users 
        WHERE flight_code IS NOT NULL AND flight_code != ''
        GROUP BY flight_code
    """)
    flights = c.fetchall()
    by_flight = {}
    for f in flights:
        if f['flight_code']:
            by_flight[f['flight_code']] = {
                'total': f['total'],
                'checked_in': f['checked_in'] or 0,
                'gate_passed': f['gate_passed'] or 0,
                'boarded': f['boarded'] or 0
            }
    
    c.execute("""
        SELECT DATE(created_at) as date, COUNT(*) as count 
        FROM users 
        WHERE created_at >= DATE('now', '-7 days')
        GROUP BY DATE(created_at)
        ORDER BY date
    """)
    daily = c.fetchall()
    daily_stats = [{'date': d['date'], 'count': d['count']} for d in daily]
    
    return jsonify({
        'total_users': total,
        'pending_users': pending,
        'checked_in': checked_in,
        'gate_passed': gate_passed,
        'boarded': boarded,
        'pending_checkin': pending_checkin,
        'approved_checkin': approved_checkin,
        'rejected_checkin': rejected_checkin,
        'by_flight': by_flight,
        'daily_stats': daily_stats
    })

@app.route('/api/admin/checkin-requests', methods=['GET'])
def get_checkin_requests():
    conn = get_db()
    c = conn.cursor()
    status = request.args.get('status')
    if status:
        c.execute("SELECT * FROM checkin_requests WHERE status=? ORDER BY created_at DESC", (status,))
    else:
        c.execute("SELECT * FROM checkin_requests ORDER BY created_at DESC")
    requests = c.fetchall()
    result = []
    for r in requests:
        data = dict(r)
        c.execute("SELECT * FROM users WHERE id=?", (data['user_id'],))
        user = c.fetchone()
        if user:
            user_dict = dict(user)
            if user_dict.get('face_image_path'):
                user_dict['face_image_url'] = f"/api/face_image/{user_dict['id']}"
            user_dict['face_crop_url'] = f"/api/face_crop/{user_dict['id']}"
            data['user'] = user_dict
        result.append(data)
    return jsonify(result)

@app.route('/api/admin/checkin-requests/<int:request_id>/review', methods=['POST'])
def review_checkin_request(request_id):
    try:
        action = request.json.get('action')
        note = request.json.get('note', '')
        
        conn = get_db()
        c = conn.cursor()
        status = 'approved' if action == 'approve' else 'rejected'
        
        if action == 'approve':
            pnr = generate_pnr()
            seat = generate_seat_number()
            gate = generate_gate()
            c.execute("""UPDATE checkin_requests SET 
                status=?, review_note=?, reviewed_at=CURRENT_TIMESTAMP,
                pnr_code=?, seat_number=?, gate_number=?
                WHERE id=?""", 
                (status, note, pnr, seat, gate, request_id))
        else:
            c.execute("UPDATE checkin_requests SET status=?, review_note=?, reviewed_at=CURRENT_TIMESTAMP WHERE id=?", 
                      (status, note, request_id))
        
        conn.commit()
        
        c.execute("SELECT user_id, flight_code, flight_date, flight_time, luggage_count, luggage_weight, pnr_code, seat_number, gate_number FROM checkin_requests WHERE id=?", (request_id,))
        row = c.fetchone()
        
        if row and action == 'approve':
            c.execute("""UPDATE users SET 
                checked_in=1, 
                flight_code=?, 
                flight_date=?, 
                flight_time=?,
                luggage_count=?,
                luggage_weight=?
                WHERE id=?""", 
                (row['flight_code'], row['flight_date'], row['flight_time'], 
                 row['luggage_count'], row['luggage_weight'], row['user_id']))
            conn.commit()
            
            c.execute("SELECT email, full_name FROM users WHERE id=?", (row['user_id'],))
            user = c.fetchone()
            
            # ⭐ Sinh PDF417 (tên hàm giữ nguyên là generate_qr_code)
            qr_path = generate_qr_code(row['user_id'], row['flight_code'], row['pnr_code'])
            if qr_path:
                c.execute("UPDATE users SET qr_code_path=? WHERE id=?", (qr_path, row['user_id']))
                conn.commit()
                
                if user and user['email']:
                    send_checkin_email(
                        to_email=user['email'],
                        passenger_name=user['full_name'],
                        flight_code=row['flight_code'],
                        flight_date=row['flight_date'],
                        flight_time=row['flight_time'],
                        pnr=row['pnr_code'],
                        seat=row['seat_number'],
                        gate=row['gate_number'],
                        qr_path=qr_path
                    )
            
            c.execute("SELECT face_image_path FROM users WHERE id=?", (row['user_id'],))
            user_data = c.fetchone()
            if user_data and user_data['face_image_path']:
                try:
                    paths = json.loads(user_data['face_image_path'])
                    if isinstance(paths, str):
                        paths = [paths]
                except:
                    paths = [user_data['face_image_path']]
                
                user_id_bg = row['user_id']
                def extract_bg(uid, p):
                    try:
                        print(f"🔍 [BG] Trích xuất vector cho user {uid} từ {len(p)} ảnh")
                        result = save_face_vector_to_db(uid, p)
                        if result:
                            print(f"✅ [BG] Xong user {uid}, vector length = {len(result)}")
                        else:
                            print(f"❌ [BG] Thất bại user {uid}")
                    except Exception as e:
                        print(f"❌ [BG] Lỗi: {e}")
                        import traceback
                        traceback.print_exc()
                
                bg_thread = threading.Thread(target=extract_bg, args=(user_id_bg, paths))
                bg_thread.daemon = True
                bg_thread.start()
        
        c.execute("INSERT INTO activity_logs (user_id, action, details) VALUES (?, ?, ?)",
                  (row['user_id'] if row else None, f'checkin_{action}', f'Admin {("duyệt" if action == "approve" else "từ chối")} yêu cầu check-in'))
        conn.commit()
        
        return jsonify({
            'success': True,
            'message': f'Đã {("duyệt" if action == "approve" else "từ chối")} yêu cầu',
            'pnr': row['pnr_code'] if row and action == 'approve' else None,
            'seat': row['seat_number'] if row and action == 'approve' else None,
            'gate': row['gate_number'] if row and action == 'approve' else None
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/checkin/request', methods=['POST'])
def request_checkin():
    try:
        user_id = request.form.get('user_id')
        flight_code = request.form.get('flight_code')
        flight_date = request.form.get('flight_date')
        flight_time = request.form.get('flight_time')
        ticket_type = request.form.get('ticket_type', 'Online có hành lý')
        luggage_count = request.form.get('luggage_count', 0)
        luggage_weight = request.form.get('luggage_weight', 0)
        
        origin = request.form.get('origin', '')
        destination = request.form.get('destination', '')
        fids_id = request.form.get('fids_id', '')
        
        face_images = request.files.getlist('face_images')
        
        if len(face_images) < 10:
            return jsonify({'error': 'Cần ít nhất 10 ảnh khuôn mặt'}), 400
        
        conn = get_db()
        c = conn.cursor()
        
        saved_paths = save_face_images(face_images, user_id)
        
        if saved_paths:
            paths_json = json.dumps(saved_paths)
            c.execute("UPDATE users SET face_image_path=? WHERE id=?", (paths_json, user_id))
            conn.commit()
            print(f"✅ Đã lưu {len(saved_paths)} ảnh cho user {user_id}")
        
        c.execute('''INSERT INTO checkin_requests 
            (user_id, flight_code, flight_date, flight_time, ticket_type, luggage_count, luggage_weight, origin, destination, fids_id, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (user_id, flight_code, flight_date, flight_time, ticket_type, luggage_count, luggage_weight, origin, destination, fids_id, 'pending'))
        
        conn.commit()
        req_id = c.lastrowid
        
        c.execute("INSERT INTO activity_logs (user_id, action, details) VALUES (?, ?, ?)",
                  (user_id, 'request_checkin', 'Gửi yêu cầu check-in thành công'))
        conn.commit()
        
        return jsonify({
            'success': True,
            'message': 'Gửi yêu cầu check-in thành công! Vui lòng chờ admin duyệt.',
            'request_id': req_id
        }), 201
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ==================== USER ====================
@app.route('/api/user/<int:user_id>/status', methods=['GET'])
def get_user_status(user_id):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id=?", (user_id,))
    user = c.fetchone()
    if not user:
        return jsonify({'error': 'Không tìm thấy user'}), 404
    
    user_dict = dict(user)
    
    c.execute("SELECT * FROM checkin_requests WHERE user_id=? ORDER BY created_at DESC", (user_id,))
    requests = c.fetchall()
    c.execute("SELECT * FROM activity_logs WHERE user_id=? ORDER BY created_at DESC LIMIT 10", (user_id,))
    logs = c.fetchall()
    
    result = {
        'user': user_dict,
        'requests': [dict(r) for r in requests],
        'logs': [dict(l) for l in logs],
        'status': {
            'pending_approval': user_dict.get('status') == 'pending',
            'checked_in': bool(user_dict.get('checked_in', 0)),
            'gate_passed': bool(user_dict.get('gate_passed', 0)),
            'boarded': bool(user_dict.get('boarded', 0))
        }
    }
    
    if user_dict.get('face_image_path'):
        result['user']['face_image_url'] = f"/api/face_image/{user_dict['id']}"
        result['user']['face_crop_url'] = f"/api/face_crop/{user_dict['id']}"
    
    return jsonify(result)

@app.route('/api/user/<int:user_id>/checkin-history', methods=['GET'])
def get_checkin_history(user_id):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM checkin_requests WHERE user_id=? ORDER BY created_at DESC", (user_id,))
    requests = c.fetchall()
    return jsonify([dict(r) for r in requests])

@app.route('/api/current_time', methods=['GET'])
def current_time():
    now = datetime.now()
    day_names = ['Thứ 2', 'Thứ 3', 'Thứ 4', 'Thứ 5', 'Thứ 6', 'Thứ 7', 'Chủ Nhật']
    day_name = day_names[now.weekday()]
    
    return jsonify({
        'time': now.strftime('%H:%M:%S'),
        'date': now.strftime('%d/%m/%Y'),
        'datetime': now.strftime('%Y-%m-%d %H:%M:%S'),
        'day': day_name
    })

# ==================== FACE VECTOR API ====================
@app.route('/api/user/<int:user_id>/face-vector', methods=['GET'])
def get_face_vector(user_id):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT face_encoding FROM users WHERE id=?", (user_id,))
    row = c.fetchone()
    if row and row['face_encoding']:
        try:
            vector = json.loads(row['face_encoding'])
            return jsonify({
                'success': True,
                'user_id': user_id,
                'face_vector': vector,
                'vector_length': len(vector),
                'algorithm': 'mobilefacenet'
            })
        except:
            return jsonify({'success': False, 'message': 'Lỗi parse vector'}), 500
    return jsonify({'success': False, 'message': 'Chưa có vector face'}), 404

@app.route('/api/user/compare-face', methods=['POST'])
def compare_face():
    try:
        data = request.json
        user_id = data.get('user_id')
        face_vector = data.get('face_vector')
        
        if not face_vector:
            return jsonify({'success': False, 'message': 'Thiếu face_vector'}), 400
        
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT face_encoding FROM users WHERE id=?", (user_id,))
        row = c.fetchone()
        
        if not row or not row['face_encoding']:
            return jsonify({'success': False, 'message': 'Không tìm thấy vector'}), 404
        
        stored_vector = np.array(json.loads(row['face_encoding']), dtype=np.float32)
        query_vector = np.array(face_vector, dtype=np.float32)
        
        stored_vector = stored_vector / np.linalg.norm(stored_vector)
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        cosine_sim = float(np.dot(stored_vector, query_vector))
        euclidean_dist = float(np.linalg.norm(stored_vector - query_vector))
        
        THRESHOLD_HIGH = 0.45
        THRESHOLD_LOW = 0.30
        EUCLIDEAN_MAX = 1.10
        
        if cosine_sim >= THRESHOLD_HIGH:
            is_match = True
            confidence = 'high'
            message = '✅ Khớp (độ tin cậy cao)'
        elif cosine_sim < THRESHOLD_LOW:
            is_match = False
            confidence = 'high'
            message = '❌ Không khớp (độ tin cậy cao)'
        else:
            if euclidean_dist < EUCLIDEAN_MAX:
                is_match = True
                confidence = 'medium'
                message = '⚠️ Có thể khớp'
            else:
                is_match = False
                confidence = 'medium'
                message = '⚠️ Có thể không khớp'
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'cosine_similarity': round(cosine_sim, 4),
            'euclidean_distance': round(euclidean_dist, 4),
            'threshold_high': THRESHOLD_HIGH,
            'threshold_low': THRESHOLD_LOW,
            'is_match': is_match,
            'confidence': confidence,
            'message': message
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ==================== FACE SEARCH 1:N CHO NODE 2 ====================
@app.route('/api/face/search', methods=['POST'])
def search_face():
    try:
        data = request.json
        face_vector = data.get('face_vector')
        threshold = float(data.get('threshold', 0.45))

        if not face_vector or not isinstance(face_vector, list):
            return jsonify({
                'success': False,
                'message': 'Thiếu face_vector hoặc sai định dạng'
            }), 400

        query_vector = np.array(face_vector, dtype=np.float32)
        if query_vector.shape[0] != 512:
            return jsonify({
                'success': False,
                'message': f'Vector phải có 512 chiều, nhận được {query_vector.shape[0]}'
            }), 400

        norm = np.linalg.norm(query_vector)
        if norm < 1e-8:
            return jsonify({
                'success': False,
                'message': 'Vector không hợp lệ (norm = 0)'
            }), 400
        query_vector = query_vector / norm

        conn = get_db()
        c = conn.cursor()
        # ⭐ SỬA: Bỏ điều kiện checked_in=1 để lấy được tất cả user đã trích xuất vector
        c.execute("""
            SELECT id, full_name, flight_code, face_encoding
            FROM users
            WHERE status='approved'
              AND face_encoding IS NOT NULL
              AND face_encoding != ''
        """)
        users = c.fetchall()

        if not users:
            return jsonify({
                'success': True,
                'matched': False,
                'message': 'Chưa có user nào được trích xuất vector',
                'candidates_count': 0,
                'top3': []
            })

        results = []
        for u in users:
            try:
                stored = np.array(json.loads(u['face_encoding']), dtype=np.float32)
                if stored.shape[0] != 512:
                    continue
                stored_norm = np.linalg.norm(stored)
                if stored_norm < 1e-8:
                    continue
                stored = stored / stored_norm
                sim = float(np.dot(stored, query_vector))
                results.append({
                    'user_id': u['id'],
                    'full_name': u['full_name'],
                    'flight_code': u['flight_code'],
                    'sim': sim
                })
            except Exception as e:
                print(f"⚠️ Lỗi parse vector user {u['id']}: {e}")
                continue

        if not results:
            return jsonify({
                'success': True,
                'matched': False,
                'message': 'Không có vector hợp lệ nào',
                'candidates_count': 0,
                'top3': []
            })

        results.sort(key=lambda x: x['sim'], reverse=True)
        top3 = results[:3]
        best = results[0]

        matched = best['sim'] >= threshold

        print(f"🔍 [FACE-SEARCH] Query vs {len(results)} users | "
              f"Best: user_id={best['user_id']} sim={best['sim']:.4f} | "
              f"threshold={threshold} | matched={matched}")

        return jsonify({
            'success': True,
            'matched': matched,
            'user_id': best['user_id'] if matched else None,
            'full_name': best['full_name'] if matched else None,
            'flight_code': best['flight_code'] if matched else None,
            'cosine_similarity': round(best['sim'], 4),
            'threshold': threshold,
            'candidates_count': len(results),
            'top3': [{'user_id': r['user_id'], 'sim': round(r['sim'], 4)} for r in top3]
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== VALIDATE PNR ====================
@app.route('/api/validate-pnr/<pnr>', methods=['GET'])
def validate_pnr(pnr):
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("""
            SELECT cr.*, u.id as user_id, u.full_name, u.flight_code, u.boarded
            FROM checkin_requests cr
            JOIN users u ON cr.user_id = u.id
            WHERE cr.pnr_code = ? AND cr.status = 'approved'
        """, (pnr,))
        row = c.fetchone()
        
        if not row:
            return jsonify({'success': False, 'message': 'PNR không hợp lệ'}), 404
        
        if row['boarded']:
            return jsonify({'success': False, 'message': 'Hành khách đã lên máy bay'}), 400
        
        return jsonify({
            'success': True,
            'user_id': row['user_id'],
            'full_name': row['full_name'],
            'flight_code': row['flight_code'],
            'seat': row['seat_number'],
            'gate': row['gate_number']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== GATE PASS API ====================
@app.route('/api/user/<int:user_id>/gate-pass', methods=['POST'])
def gate_pass(user_id):
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE id=?", (user_id,))
        user = c.fetchone()
        if not user:
            return jsonify({'error': 'Không tìm thấy user'}), 404
        if not user['checked_in']:
            return jsonify({'error': 'Hành khách chưa check-in'}), 400
        if user['gate_passed']:
            return jsonify({'error': 'Hành khách đã qua cổng'}), 400
        
        c.execute("UPDATE users SET gate_passed=1 WHERE id=?", (user_id,))
        conn.commit()
        c.execute("INSERT INTO activity_logs (user_id, action, details) VALUES (?, ?, ?)",
                  (user_id, 'gate_pass', 'Qua cổng thành công'))
        conn.commit()
        
        return jsonify({
            'success': True,
            'message': 'Xác nhận qua cổng thành công',
            'user': dict(user)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== BOARDING API ====================
@app.route('/api/user/<int:user_id>/board', methods=['POST'])
def board(user_id):
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE id=?", (user_id,))
        user = c.fetchone()
        if not user:
            return jsonify({'error': 'Không tìm thấy user'}), 404
        if not user['gate_passed']:
            return jsonify({'error': 'Hành khách chưa qua cổng'}), 400
        if user['boarded']:
            return jsonify({'error': 'Hành khách đã lên máy bay'}), 400
        
        c.execute("UPDATE users SET boarded=1 WHERE id=?", (user_id,))
        conn.commit()
        c.execute("INSERT INTO activity_logs (user_id, action, details) VALUES (?, ?, ?)",
                  (user_id, 'board', 'Lên máy bay thành công'))
        conn.commit()
        
        return jsonify({
            'success': True,
            'message': 'Xác nhận lên máy bay thành công',
            'user': dict(user)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Server đang chạy...")
    
    # ⭐ TỰ ĐỘNG GỌI FILE OLED CHẠY SONG SONG
    try:
        from oled_display import run_oled
        oled_thread = threading.Thread(target=run_oled, daemon=True)
        oled_thread.start()
    except ImportError:
        print("⚠️ Không tìm thấy file oled_display.py")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)