# app/api_client.py
import requests
import json
from app.config import Config

class APIClient:
    def __init__(self):
        self.base_url = Config.SERVER_URL
        self.timeout = 30
    
    def _request(self, method, endpoint, data=None):
        url = f"{self.base_url}{endpoint}"
        try:
            if method == 'GET':
                response = requests.get(url, timeout=self.timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, timeout=self.timeout)
            elif method == 'PUT':
                response = requests.put(url, json=data, timeout=self.timeout)
            elif method == 'DELETE':
                response = requests.delete(url, timeout=self.timeout)
            else:
                return {'error': 'Method not supported'}, 405
            return response.json(), response.status_code
        except Exception as e:
            return {'error': str(e)}, 500
    
    def register(self, username, password, email, phone, full_name, cccd):
        url = f"{self.base_url}/auth/register"
        data = {
            'username': username,
            'password': password,
            'email': email,
            'phone': phone,
            'full_name': full_name,
            'cccd': cccd
        }
        try:
            response = requests.post(url, json=data, timeout=self.timeout)
            return response.json(), response.status_code
        except Exception as e:
            return {'error': str(e)}, 500
    
    def login(self, username, password):
        url = f"{self.base_url}/auth/login"
        data = {'username': username, 'password': password}
        try:
            response = requests.post(url, json=data, timeout=self.timeout)
            return response.json(), response.status_code
        except Exception as e:
            return {'error': str(e)}, 500
    
    # ⭐ SỬA: Thêm origin, destination, fids_id
    def request_checkin(self, user_id, flight_code, flight_date, flight_time, 
                        ticket_type, luggage_count, luggage_weight, face_images_base64,
                        origin='', destination='', fids_id=''):
        url = f"{self.base_url}/checkin/request"
        
        data = {
            'user_id': user_id,
            'flight_code': flight_code,
            'flight_date': flight_date,
            'flight_time': flight_time,
            'ticket_type': ticket_type,
            'luggage_count': luggage_count,
            'luggage_weight': luggage_weight,
            'origin': origin,
            'destination': destination,
            'fids_id': fids_id
        }
        
        import base64
        files = []
        for idx, img_base64 in enumerate(face_images_base64):
            img_bytes = base64.b64decode(img_base64)
            files.append(('face_images', (f'face_{idx}.jpg', img_bytes, 'image/jpeg')))
        
        try:
            response = requests.post(url, data=data, files=files, timeout=self.timeout)
            return response.json(), response.status_code
        except Exception as e:
            return {'error': str(e)}, 500
    
    def get_users(self):
        url = f"{self.base_url}/admin/users"
        try:
            response = requests.get(url, timeout=self.timeout)
            return response.json(), response.status_code
        except Exception as e:
            return {'error': str(e)}, 500
    
    def approve_user(self, user_id, action):
        url = f"{self.base_url}/admin/users/{user_id}/approve"
        data = {'action': action}
        try:
            response = requests.post(url, json=data, timeout=self.timeout)
            return response.json(), response.status_code
        except Exception as e:
            return {'error': str(e)}, 500
    
    def get_stats(self):
        url = f"{self.base_url}/admin/stats"
        try:
            response = requests.get(url, timeout=self.timeout)
            return response.json(), response.status_code
        except Exception as e:
            return {'error': str(e)}, 500
    
    def get_checkin_requests(self, status=None):
        url = f"{self.base_url}/admin/checkin-requests"
        if status:
            url += f"?status={status}"
        try:
            response = requests.get(url, timeout=self.timeout)
            return response.json(), response.status_code
        except Exception as e:
            return {'error': str(e)}, 500
    
    def review_checkin_request(self, request_id, action, note='', admin_id=1):
        url = f"{self.base_url}/admin/checkin-requests/{request_id}/review"
        data = {
            'action': action,
            'note': note,
            'admin_id': admin_id
        }
        try:
            response = requests.post(url, json=data, timeout=self.timeout)
            return response.json(), response.status_code
        except Exception as e:
            return {'error': str(e)}, 500

    # ==================== FIDS METHODS ====================
    def get_fids(self):
        return self._request('GET', '/fids')
    
    def get_fids_detail(self, fids_id):
        return self._request('GET', f'/fids/{fids_id}')
    
    def create_fids(self, data):
        return self._request('POST', '/fids', data)
    
    def update_fids(self, fids_id, data):
        return self._request('PUT', f'/fids/{fids_id}', data)
    
    def delete_fids(self, fids_id):
        return self._request('DELETE', f'/fids/{fids_id}')

    # ==================== USER STATUS METHODS ====================
    def get_user_status(self, user_id):
        return self._request('GET', f'/user/{user_id}/status')
    
    def get_checkin_history(self, user_id):
        return self._request('GET', f'/user/{user_id}/checkin-history')

    # ==================== FACE VECTOR METHODS ====================
    def get_face_vector(self, user_id):
        return self._request('GET', f'/user/{user_id}/face-vector')
    
    def compare_face(self, data):
        return self._request('POST', '/user/compare-face', data)
    
    def gate_pass(self, user_id):
        return self._request('POST', f'/user/{user_id}/gate-pass')
    
    def board(self, user_id):
        return self._request('POST', f'/user/{user_id}/board')
