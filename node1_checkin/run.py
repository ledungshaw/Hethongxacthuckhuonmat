# run.py
import os
from flask import Flask
from flask_cors import CORS
from app.config import Config
from app.routes import init_routes

os.environ['TZ'] = 'Asia/Ho_Chi_Minh'

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = Config.SECRET_KEY
app.config['DEBUG'] = Config.DEBUG
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

CORS(app)

# Khởi tạo routes
init_routes(app)

if __name__ == '__main__':
    os.makedirs('uploads/temp', exist_ok=True)
    os.makedirs('database', exist_ok=True)
    app.run(host='0.0.0.0', port=8080, debug=True)
