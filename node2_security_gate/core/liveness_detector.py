"""
Liveness Detector - Challenge chop mat bang OpenCV Haar Eye Detector
Khong can MediaPipe - chay tot tren Pi 4-8GB.
Logic: mat mo -> detect 2 mat | mat nham -> detect 0 mat
"""
import cv2
import numpy as np
import time
from utils.logger import get_logger

logger = get_logger(__name__)


class LivenessDetector:
    """
    Phat hien chop mat bang Haar Cascade Eye Detector (OpenCV).
    Yeu cau user chop mat it nhat N lan trong timeout giay.
    """

    def __init__(self, min_eye_size=(20, 20)):
        self.min_eye_size = min_eye_size
        self.available = False

        try:
            # Load Haar Cascade cho mat
            eye_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            self.eye_cascade = cv2.CascadeClassifier(eye_path)
            if self.eye_cascade.empty():
                raise Exception(f"Khong load duoc: {eye_path}")

            # Load Haar Cascade cho mat nguoi (de cat vung mat)
            face_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(face_path)
            if self.face_cascade.empty():
                raise Exception(f"Khong load duoc: {face_path}")

            self.available = True
            logger.info("✅ LivenessDetector (OpenCV Haar Eye) loaded")
            print("✅ Node 2 LivenessDetector (OpenCV Haar Eye) loaded")
        except Exception as e:
            logger.error(f"❌ Khong load duoc Haar Cascade: {e}")
            print(f"⚠️ Khong load duoc Haar Cascade: {e}")
            self.available = False

    def _detect_eyes(self, face_upper_bgr):
        """Tra ve so luong mat phat hien duoc trong vung mat (nua tren)"""
        if face_upper_bgr is None or face_upper_bgr.size == 0:
            return 0
        gray = cv2.cvtColor(face_upper_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.min_eye_size
        )
        return len(eyes)

    def _detect_face(self, frame_bgr):
        """Tra ve box mat lon nhat hoac None"""
        if frame_bgr is None:
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            return None
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return faces[0]

    def check_blink(self, camera, timeout=8.0, required_blinks=1):
        """
        Yeu cau user chop mat it nhat `required_blinks` lan trong `timeout` giay.
        Dung Haar Eye: mat mo = detect 2 mat, mat nham = detect 0 mat.

        Tra ve: {is_live, blink_count, elapsed, message}
        """
        if not self.available:
            return {
                'is_live': True,
                'blink_count': 0,
                'elapsed': 0.0,
                'message': '⚠️ Liveness OFF (khong co Haar Eye)'
            }

        blink_count = 0
        eyes_closed_state = False
        start = time.time()
        frame_count = 0

        print(f"👁️ [Liveness] Bat dau kiem tra chop mat ({required_blinks} lan trong {timeout}s)...")

        while time.time() - start < timeout:
            frame = camera.capture_frame()
            if frame is None:
                continue

            frame_count += 1

            # Detect face
            face_box = self._detect_face(frame)

            n_eyes = 0
            if face_box is not None:
                x, y, w, h = face_box
                # Cat nua tren cua face (vung mat)
                face_upper = frame[y:y + h // 2, x:x + w]
                n_eyes = self._detect_eyes(face_upper)

            # Logic phat hien chop mat
            if n_eyes >= 2:
                # Mat mo -> neu truoc do dang nham thi tinh 1 lan chop
                if eyes_closed_state:
                    blink_count += 1
                    print(f"   👁️ Phat hien chop mat lan {blink_count}")
                    eyes_closed_state = False
                    if blink_count >= required_blinks:
                        # Ve khung xanh thong bao thanh cong
                        display = frame.copy()
                        cv2.putText(display, "✅ DA CHOP MAT - OK",
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 255, 0), 2)
                        cv2.imshow('Node 2 - Security Gate', display)
                        cv2.waitKey(300)
                        break
            elif n_eyes == 0 and face_box is not None:
                # Mat nham
                eyes_closed_state = True

            # Hien thi huong dan
            display = frame.copy()
            if face_box is not None:
                x, y, w, h = face_box
                cv2.rectangle(display, (x, y), (x + w, y + h), (255, 255, 0), 2)

            cv2.putText(display, "HAY CHOP MAT DE XAC THUC NGUOI THAT",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 255), 2)
            cv2.putText(display, f"Blinks: {blink_count}/{required_blinks} | Eyes: {n_eyes}",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            time_left = timeout - (time.time() - start)
            cv2.putText(display, f"Time: {time_left:.1f}s",
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 200, 255), 2)

            cv2.imshow('Node 2 - Security Gate', display)
            cv2.waitKey(1)

        elapsed = time.time() - start
        is_live = blink_count >= required_blinks

        return {
            'is_live': is_live,
            'blink_count': blink_count,
            'elapsed': round(elapsed, 2),
            'message': (
                f'✅ Da chop mat {blink_count} lan'
                if is_live else
                f'❌ Khong phat hien chop mat ({blink_count}/{required_blinks})'
            )
        }