import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sync_manager import SyncManager
from config import Config

def test_api():
    print("="*50)
    print("🔍 TEST API CONNECTION")
    print("="*50)
    
    sync = SyncManager()
    print(f"\n📡 Server: {Config.SERVER_URL}")
    
    if sync.test_connection():
        print("✅ Server connection OK")
    else:
        print("❌ Server connection FAILED")
        return
    
    print("\n🔄 Syncing faces...")
    faces = sync.sync_from_server(force=True)
    print(f"✅ Loaded {len(faces)} faces")
    
    for f in faces[:5]:
        print(f"   - ID: {f['user_id']} - {f['name']}")

if __name__ == "__main__":
    test_api()
