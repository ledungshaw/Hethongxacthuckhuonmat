import cv2
import numpy as np
import onnxruntime as ort
import pickle
from core.face_detector import FaceDetectorMTCNN
from core.camera_handler import CameraHandler

print("="*50)
print("🧪 TEST SIMILARITY")
print("="*50)

# Load MobileFaceNet
session = ort.InferenceSession('models/mobilefacenet.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"✅ MobileFaceNet loaded")

# Load cache
with open('dataset/encodings/encodings.pkl', 'rb') as f:
    data = pickle.load(f)
print(f"✅ Loaded {len(data)} faces from cache")

def preprocess(img):
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)

def extract(img):
    inp = preprocess(img)
    out = session.run([output_name], {input_name: inp})
    emb = out[0][0]
    return emb / np.linalg.norm(emb)

# Camera
camera = CameraHandler()
detector = FaceDetectorMTCNN()

print("\n📸 Press SPACE to capture, ESC to exit")

while True:
    frame = camera.capture_frame()
    if frame is None:
        continue
    
    faces = detector.detect(frame)
    
    display = frame.copy()
    if faces:
        for (top, right, bottom, left) in faces:
            cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(display, "Press SPACE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Test Similarity', display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32 and faces:  # SPACE
        top, right, bottom, left = faces[0]
        face_img = frame[top:bottom, left:right]
        
        # Extract vector
        new_emb = extract(face_img)
        
        print("\n" + "="*50)
        print("📊 SIMILARITY RESULTS")
        print("="*50)
        
        for face in data:
            stored = np.array(face['encoding'])
            sim = float(np.dot(stored, new_emb) / (np.linalg.norm(stored) * np.linalg.norm(new_emb)))
            
            # Phân loại
            if sim > 0.6:
                status = "✅ MATCH"
            elif sim > 0.4:
                status = "⚠️ CÓ THỂ"
            else:
                status = "❌ KHÔNG"
            
            print(f"   {face['name']}: {sim:.4f} {status}")
        
        print("\n📌 Ngưỡng hiện tại: 0.6")
        print("="*50 + "\n")

camera.release()
cv2.destroyAllWindows()
