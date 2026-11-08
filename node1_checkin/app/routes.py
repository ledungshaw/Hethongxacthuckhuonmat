# app/routes.py
from flask import render_template, request, jsonify, session, redirect, url_for
from app.api_client import APIClient
from app.camera_handler import CameraHandler
import base64
import os
import cv2
from datetime import datetime

api = APIClient()
camera = CameraHandler()

def init_routes(app):
    
    @app.route('/')
    def index():
        if 'user_id' in session:
            if session.get('role') == 'admin':
                return redirect('/admin/dashboard')
            else:
                return redirect('/user/dashboard')
        return render_template('index.html')
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            
            result, status = api.login(username, password)
            
            if status == 200 and result.get('success'):
                session['user_id'] = result['user']['id']
                session['role'] = result['user']['role']
                session['username'] = result['user']['username']
                session['full_name'] = result['user']['full_name']
                
                if result['user']['role'] == 'admin':
                    return redirect('/admin/dashboard')
                else:
                    return redirect('/user/dashboard')
            else:
                return render_template('login.html', error=result.get('error', 'Đăng nhập thất bại'))
        
        return render_template('login.html')
    
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            email = request.form.get('email')
            phone = request.form.get('phone')
            full_name = request.form.get('full_name')
            cccd = request.form.get('cccd')
            
            if password != confirm_password:
                return render_template('register.html', error='Mật khẩu không khớp')
            
            result, status = api.register(username, password, email, phone, full_name, cccd)
            
            if status == 201 and result.get('success'):
                return render_template('register.html', 
                                     success='Đăng ký thành công! Vui lòng chờ admin duyệt tài khoản.')
            else:
                return render_template('register.html', error=result.get('error', 'Đăng ký thất bại'))
        
        return render_template('register.html')
    
    @app.route('/logout')
    def logout():
        session.clear()
        return redirect('/')
    
    @app.route('/user/dashboard')
    def user_dashboard():
        if 'user_id' not in session:
            return redirect('/login')
        return render_template('user_dashboard.html', 
                             username=session.get('username'),
                             full_name=session.get('full_name'),
                             user_id=session.get('user_id'))
    
    @app.route('/user/checkin')
    def user_checkin():
        if 'user_id' not in session:
            return redirect('/login')
        return render_template('user_checkin.html', 
                             user_id=session.get('user_id'),
                             username=session.get('username'))
    
    @app.route('/admin/dashboard')
    def admin_dashboard():
        if 'user_id' not in session or session.get('role') != 'admin':
            return redirect('/login')
        return render_template('admin_dashboard.html', 
                             username=session.get('username'))
    
    @app.route('/admin/checkin')
    def admin_checkin():
        if 'user_id' not in session or session.get('role') != 'admin':
            return redirect('/login')
        return render_template('admin_checkin.html', 
                             username=session.get('username'))
    
    @app.route('/admin/requests')
    def admin_requests():
        if 'user_id' not in session or session.get('role') != 'admin':
            return redirect('/login')
        return render_template('admin_requests.html', 
                             username=session.get('username'))
    
    @app.route('/api/current_time')
    def current_time():
        now = datetime.now()
        return jsonify({
            'time': now.strftime('%H:%M:%S'),
            'date': now.strftime('%d/%m/%Y'),
            'day': ['Chủ Nhật','Thứ 2','Thứ 3','Thứ 4','Thứ 5','Thứ 6','Thứ 7'][now.weekday()]
        })
    
    @app.route('/api/capture', methods=['POST'])
    def capture_photo():
        try:
            frame = camera.capture_frame()
            if frame is None:
                return jsonify({'error': 'Không thể chụp ảnh'}), 500
            
            faces = camera.detect_faces(frame)
            if len(faces) == 0:
                return jsonify({'error': 'Không tìm thấy khuôn mặt'}), 400
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            image_base64 = camera.frame_to_base64(frame)
            return jsonify({
                'success': True,
                'image': image_base64,
                'faces': len(faces)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/checkin', methods=['POST'])
    def admin_checkin_api():
        try:
            name = request.form.get('name')
            cccd = request.form.get('cccd')
            phone = request.form.get('phone')
            email = request.form.get('email')
            flight_code = request.form.get('flight_code')
            flight_date = request.form.get('flight_date')
            flight_time = request.form.get('flight_time')
            ticket_type = request.form.get('ticket_type', 'Online có hành lý')
            luggage_count = request.form.get('luggage_count', 0)
            luggage_weight = request.form.get('luggage_weight', 0)
            
            face_images = request.files.getlist('face_images')
            
            if len(face_images) < 3:
                return jsonify({'error': 'Cần ít nhất 3 ảnh khuôn mặt'}), 400
            
            face_images_base64 = []
            for img in face_images:
                img_bytes = img.read()
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                face_images_base64.append(img_base64)
            
            import random
            temp_username = f"guest_{random.randint(1000, 9999)}"
            temp_password = "guest123"
            
            result, status = api.register(
                temp_username, temp_password, email, phone, name, cccd
            )
            
            if status != 201 or not result.get('success'):
                return jsonify({'error': 'Không thể tạo tài khoản'}), 500
            
            user_id = result.get('user_id')
            
            result2, status2 = api.request_checkin(
                user_id, flight_code, flight_date, flight_time,
                ticket_type, luggage_count, luggage_weight, face_images_base64
            )
            
            return jsonify(result2), status2
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    # ⭐ SỬA: Nhận thêm origin, destination, fids_id
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
            
            # ⭐ THÊM
            origin = request.form.get('origin', '')
            destination = request.form.get('destination', '')
            fids_id = request.form.get('fids_id', '')
            
            face_images = request.files.getlist('face_images')
            
            if len(face_images) < 3:
                return jsonify({'error': 'Cần ít nhất 3 ảnh khuôn mặt'}), 400
            
            face_images_base64 = []
            for img in face_images:
                img_bytes = img.read()
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                face_images_base64.append(img_base64)
            
            # ⭐ Truyền thêm origin, destination, fids_id
            result, status = api.request_checkin(
                user_id, flight_code, flight_date, flight_time,
                ticket_type, luggage_count, luggage_weight, face_images_base64,
                origin=origin, destination=destination, fids_id=fids_id
            )
            
            return jsonify(result), status
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/admin/stats')
    def admin_stats():
        try:
            result, status = api.get_stats()
            return jsonify(result), status
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/admin/users')
    def admin_users():
        try:
            result, status = api.get_users()
            return jsonify(result), status
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/admin/users/<int:user_id>/approve', methods=['POST'])
    def admin_approve_user(user_id):
        try:
            action = request.json.get('action')
            result, status = api.approve_user(user_id, action)
            return jsonify(result), status
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/admin/checkin-requests')
    def admin_checkin_requests():
        try:
            status_filter = request.args.get('status')
            result, status = api.get_checkin_requests(status_filter)
            return jsonify(result), status
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/admin/checkin-requests/<int:request_id>/review', methods=['POST'])
    def admin_review_checkin(request_id):
        try:
            action = request.json.get('action')
            note = request.json.get('note', '')
            admin_id = session.get('user_id', 1)
            result, status = api.review_checkin_request(request_id, action, note, admin_id)
            return jsonify(result), status
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # ==================== FIDS API ====================
    @app.route('/api/fids', methods=['GET'])
    def get_fids():
        try:
            result, status = api.get_fids()
            return jsonify(result), status
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/fids', methods=['POST'])
    def create_fids():
        try:
            data = request.json
            result, status = api.create_fids(data)
            return jsonify(result), status
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/fids/<int:fids_id>', methods=['GET'])
    def get_fids_detail(fids_id):
        try:
            result, status = api.get_fids_detail(fids_id)
            return jsonify(result), status
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/fids/<int:fids_id>', methods=['PUT'])
    def update_fids(fids_id):
        try:
            data = request.json
            result, status = api.update_fids(fids_id, data)
            return jsonify(result), status
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/fids/<int:fids_id>', methods=['DELETE'])
    def delete_fids(fids_id):
        try:
            result, status = api.delete_fids(fids_id)
            return jsonify(result), status
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # ==================== USER STATUS API ====================
    @app.route('/api/user/<int:user_id>/status', methods=['GET'])
    def user_status(user_id):
        try:
            result, status = api.get_user_status(user_id)
            return jsonify(result), status
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    @app.route('/api/user/<int:user_id>/checkin-history', methods=['GET'])
    def user_checkin_history(user_id):
        try:
            result, status = api.get_checkin_history(user_id)
            return jsonify(result), status
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # ==================== FACE VECTOR API ====================
    @app.route('/api/user/<int:user_id>/face-vector', methods=['GET'])
    def get_face_vector(user_id):
        try:
            result, status = api.get_face_vector(user_id)
            return jsonify(result), status
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/user/compare-face', methods=['POST'])
    def compare_face():
        try:
            data = request.json
            result, status = api.compare_face(data)
            return jsonify(result), status
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/user/<int:user_id>/gate-pass', methods=['POST'])
    def gate_pass(user_id):
        try:
            result, status = api.gate_pass(user_id)
            return jsonify(result), status
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/user/<int:user_id>/board', methods=['POST'])
    def board(user_id):
        try:
            result, status = api.board(user_id)
            return jsonify(result), status
        except Exception as e:
            return jsonify({'error': str(e)}), 500
