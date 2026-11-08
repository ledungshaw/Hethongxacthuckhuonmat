"""
Camera Handler (CSI - Picamera2) - Bản tối ưu FOV + Độ nét
- Cung cấp frame gốc cho AI
"""
import threading
import time
import cv2
from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

class CameraHandler:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.lock = threading.Lock()
        self._started = False

    def start_camera(self):
        try:
            from picamera2 import Picamera2
            from libcamera import ColorSpace, Transform

            if self.cap is not None:
                try: self.cap.stop()
                except: pass
                try: self.cap.close()
                except: pass
                self.cap = None

            self.cap = Picamera2()

            # ⭐ SỬA: Quay lại 1280x960 để Camera chạy native, không bị giật
            config = self.cap.create_video_configuration(
                main={"size": (1280, 960), "format": "RGB888"},
                transform=Transform(hflip=False, vflip=False),
                colour_space=ColorSpace.Sycc(),
                buffer_count=4
            )
            self.cap.configure(config)
            
            self.cap.start()
            time.sleep(1.5)
            
            try: 
                self.cap.set_controls({
                    "AwbEnable": True, 
                    "AeEnable": True,
                    "Sharpness": 1.5,   # Giảm nhẹ độ nét để đỡ tốn CPU
                    "Contrast": 1.1
                })
            except: pass
            time.sleep(1.0)

            self._started = True
            print(f"✅ Camera CSI: 1280x960 (Native Video Mode)")
            return True
        except Exception as e:
            logger.error(f"❌ Camera start failed: {e}")
            return False

    def capture_frame(self):
        with self.lock:
            if self.cap is None or not self._started:
                if not self.start_camera():
                    return None
            try:
                frame = self.cap.capture_array("main")
                return frame if frame.shape[2] == 3 else frame[:, :, :3]
            except Exception:
                return None

    def release(self):
        with self.lock:
            if self.cap:
                try: self.cap.stop()
                except: pass
                try: self.cap.close()
                except: pass
                self.cap = None
                self._started = False

    def __del__(self):
        self.release()