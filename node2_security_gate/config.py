# config.py - Cau hinh Node 2
import os


class Config:
    SERVER_URL = 'http://192.168.137.13:5000/api'

    DATASET_PATH = 'dataset/known_faces'
    ENCODINGS_PATH = 'dataset/encodings/encodings.pkl'

    # ===== CAMERA (Camera Module V2 / IMX219) =====
    CAMERA_RESOLUTION = (1280, 960)
    CAMERA_SENSOR_SIZE = (1640, 1232)
    CAMERA_FPS = 30
    CAMERA_BUFFER_COUNT = 4
    CAMERA_BUFFER_SIZE = 1
    CAMERA_WARMUP_FRAMES = 30

    CAMERA_AWB_MODE = "Auto"
    CAMERA_NOISE_REDUCTION = "Fast"
    CAMERA_SHARPNESS = 1.0
    CAMERA_CONTRAST = 1.0
    CAMERA_SATURATION = 1.0
    CAMERA_BRIGHTNESS = 0.0
    CAMERA_HFLIP = False
    CAMERA_VFLIP = False

    # ===== STREAM WEB (MJPEG) =====
    STREAM_JPEG_QUALITY = 95
    STREAM_JPEG_444 = True
    STREAM_MAX_FPS = 30
    OVERLAY_TTL = 1.0

    # ===== TOI UU TOC DO =====
    PROCESS_EVERY_N_FRAMES = 6
    FACE_LOST_TTL = 30
    AI_THREAD_SLEEP = 0.01

    # ===== FACE RECOGNITION =====
    THRESHOLD = 0.48
    THRESHOLD_LOW = 0.30
    MIN_FACE_SIZE = (80, 80)

    # ===== FACE DETECTOR =====
    HAAR_SCALE_FACTOR = 1.15
    HAAR_MIN_NEIGHBORS = 6
    USE_MTCNN = True

    MTCNN_MIN_FACE_SIZE = 60
    MTCNN_FACTOR = 0.8
    MTCNN_THRESHOLDS = [0.6, 0.7, 0.7]

    # ===== TRACKING / SMOOTHING =====
    SMOOTH_FACTOR = 0.7

    # ===== ANTI-SPOOFING =====
    ANTI_SPOOF_INPUT_SIZE = (128, 128)
    ANTI_SPOOFING_INTERVAL = 10

    # ⭐ Ngưỡng chuẩn cho Ensemble Multi-Scale MiniFASNet (đã chuẩn hóa dải pixel 0-255)
    LIVENESS_THRESHOLD = 0.60
    LIVENESS_REJECT = 0.30
    LIVENESS_CACHE_TTL = 0.3
    LIVENESS_SMOOTHING_WINDOW = 5

    # ===== LIVENESS CHALLENGE =====
    ENABLE_LIVENESS_CHALLENGE = False
    LIVENESS_BLINK_REQUIRED = 1
    LIVENESS_TIMEOUT = 3.0
    HAAR_EYE_MIN_SIZE = (20, 20)
    LIVENESS_FRAME_SKIP = 2

    # ===== ONNX =====
    ONNX_INTRA_THREADS = 4
    ONNX_INTER_THREADS = 2

    # ===== MQTT =====
    ENABLE_MQTT = False
    MQTT_BROKER = "localhost"
    MQTT_PORT = 1883
    MQTT_TOPIC_USER_NEW = "airport/user/new"

    # ===== LOGGING =====
    LOG_LEVEL = 'WARNING'
    LOG_FILE = 'logs/node2.log'

    FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    FONT_SIZE = 20

    MODEL_DIR = 'models'
    MOBILEFACENET_PATH = 'models/mobilefacenet_new.onnx'
    MINIFASNET_V2_PATH = 'models/minifasnetv2_80x80.onnx'
    MINIFASNET_V1SE_PATH = 'models/minifasnetv1se_80x80.onnx'
    MINIFASNET_PATH = 'models/minifasnetv2.onnx'


def _ensure_directories():
    os.makedirs(Config.DATASET_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(Config.ENCODINGS_PATH), exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)


_ensure_directories()