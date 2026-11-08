import time
import sqlite3
import os

# Cấu hình database
DB_PATH = 'database/airport.db'
I2C_ADDRESS = 0x3C  # Đổi thành 0x3D nếu cần

def run_oled():
    """Hàm chạy hiển thị OLED (sẽ được gọi từ server.py)"""
    try:
        from luma.core.interface.serial import i2c
        from luma.oled.device import ssd1306
        from luma.core.render import canvas
        from PIL import ImageFont
    except ImportError:
        print("⚠️ Chưa cài luma.oled. Bỏ qua hiển thị OLED.")
        return

    try:
        # Khởi tạo màn hình
        serial = i2c(port=1, address=I2C_ADDRESS)
        device = ssd1306(serial, width=128, height=64)
        
        # ⭐ GIẢM FONT CHỮ XUỐNG NHỎ HƠN NỮA
        try:
            # Tiêu đề: size 8
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 8)
            # Nội dung: size 7 (nhỏ nhất có thể đọc được)
            font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 7)
        except:
            font_title = ImageFont.load_default()
            font_text = ImageFont.load_default()
            
        print(f"✅ Đã khởi tạo màn hình OLED (128x64) tại địa chỉ {hex(I2C_ADDRESS)}")
        
        while True:
            try:
                if not os.path.exists(DB_PATH):
                    time.sleep(2)
                    continue
                    
                conn = sqlite3.connect(DB_PATH, timeout=5)
                c = conn.cursor()
                c.execute("""
                    SELECT flight_code, origin, destination, scheduled_time, gate 
                    FROM fids 
                    WHERE status IN ('Scheduled', 'Boarding')
                    ORDER BY flight_date, scheduled_time LIMIT 3
                """)
                flights = c.fetchall()
                conn.close()
                
                # Vẽ lên màn hình
                with canvas(device) as draw:
                    # Tiêu đề (Căn giữa)
                    title = "✈ FLIGHT INFO ✈"
                    try:
                        bbox = draw.textbbox((0, 0), title, font=font_title)
                        title_w = bbox[2] - bbox[0]
                    except:
                        title_w = len(title) * 5
                    draw.text(((128 - title_w) // 2, 0), title, font=font_title, fill="white")
                    
                    # Đường kẻ dưới tiêu đề
                    draw.line((0, 10, 128, 10), fill="white")
                    
                    if not flights:
                        draw.text((30, 30), "No flights", font=font_text, fill="white")
                    else:
                        y_start = 13
                        row_height = 16 # Khoảng cách giữa các chuyến
                        
                        for i, f in enumerate(flights):
                            code, origin, dest, time_str, gate = f
                            y = y_start + (i * row_height)
                            
                            # Dòng 1: Mã chuyến (Trái) | Giờ (Giữa) | GATE (Phải)
                            draw.text((0, y), str(code), font=font_text, fill="white")
                            
                            # Giờ (căn giữa ở vị trí 50)
                            draw.text((50, y), str(time_str), font=font_text, fill="white")
                            
                            # GATE (căn phải)
                            if gate:
                                gate_text = f"GATE {gate}"
                                try:
                                    bbox_gate = draw.textbbox((0, 0), gate_text, font=font_text)
                                    gate_w = bbox_gate[2] - bbox_gate[0]
                                except:
                                    gate_w = len(gate_text) * 5
                                draw.text((128 - gate_w, y), gate_text, font=font_text, fill="white")
                            
                            # Dòng 2: Hành trình
                            origin_short = origin[:6] if origin else "---"
                            dest_short = dest[:6] if dest else "---"
                            route = f"{origin_short} -> {dest_short}"
                            draw.text((0, y + 8), route, font=font_text, fill="white")
                            
                            # Đường kẻ phân cách
                            if i < len(flights) - 1:
                                draw.line((0, y + 15, 128, y + 15), fill="white", width=1)

            except Exception as e:
                pass
            
            time.sleep(5)
            
    except Exception as e:
        print(f"❌ Không thể khởi tạo OLED: {e}")

if __name__ == "__main__":
    print("Chạy trực tiếp file OLED...")
    run_oled()