from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from datetime import datetime
from pymongo import MongoClient
from bson.objectid import ObjectId
import certifi
import base64
import gridfs
import traceback

app = Flask(__name__)

# --- 1. まず最初に socketio を定義する ---
# max_http_buffer_size を大きくして画像送信を許可します
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=20000000)

# MongoDB設定
mongo_uri = "mongodb+srv://e221239_db_user:Y0u667841you@cluster0.1tmhycc.mongodb.net/?appName=Cluster0"
# 後ろにカンマを打って、serverSelectionTimeoutMS を追加する
client = MongoClient(mongo_uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=20000)
db = client["SNS"]
fs = gridfs.GridFS(db)

@app.route("/")
def index():
    return render_template("index.html")

def get_image_data(image_id):
    try:
        if not fs.exists(image_id):
            return ""
        image_file = fs.get(image_id).read()
        # base64エンコードして文字列で返す
        return base64.b64encode(image_file).decode('utf-8')
    except Exception as e:
        print(f"Error: {e}")
        return ""

# --- 2. socketio 定義の後にデコレータを書く ---
@socketio.on('load messages')
def load_messages():
    try:
        # DBから最新のメッセージを取得
        messages = db.images.find().sort('_id', -1).limit(50) # 件数は適宜
        messages = list(messages)[::-1]
        
        messages_return = []
        for msg in messages:
            img_b64 = get_image_data(msg['image_id'])
            
            # --- ここが重要だッ！ project_name を辞書に含めて返せッ！ ---
            messages_return.append({
                '_id': str(msg['_id']),
                'project_name': msg.get('project_name', '未分類'), # これを忘れてたんだッ！
                'material_name': msg.get('material_name', '名称不明'),
                'message': msg.get('message', ''),
                'image_data': f"data:image/jpeg;base64,{img_b64}" if img_b64 else ""
            })
            
        emit('load all messages', messages_return)
        
    except Exception as e:
        traceback.print_exc()

@socketio.on('send message')
def send_message(data):
    try:
        project_name = data.get('project_name', '未分類') 
        material_name = data.get('material_name', '')
        message = data.get('message', '')
        image_data = data.get('image_data', '')

        if image_data and ',' in image_data:
            encoded = image_data.split(',', 1)[1]
            image_bytes = base64.b64decode(encoded)
            image_id = fs.put(image_bytes, filename="upload.jpg")

            # 保存して、そのドキュメントの情報を取得するッ！
            result = db.images.insert_one({
                'project_name': project_name,
                'material_name': material_name,
                'image_id': image_id,
                'message': message,
                'date': datetime.now()
            })

            # クライアント全員に、確定した『_id』を添えて送り返すんだッ！！
            emit('load one message', {
                '_id': str(result.inserted_id), # これが重要だッ！
                "project_name": project_name,
                "material_name": material_name,
                "message": message,
                "image_data": image_data
            }, broadcast=True)
    except Exception as e:
        traceback.print_exc()
@socketio.on('delete message')
def delete_message(data):
    try:
        msg_id = data.get('id')
        if msg_id:
            # MongoDBから削除する（画像も消すならGridFSの処理も入れるぜ）
            # まずはドキュメントを特定して削除だッ！
            target = db.images.find_one({"_id": ObjectId(msg_id)})
            if target:
                # 画像(GridFS)も一緒に消し去るッ！
                fs.delete(target['image_id'])
                # 本体も消去ッ！
                db.images.delete_one({"_id": ObjectId(msg_id)})
                
                # 全員に「消したぞ」と通知して画面を更新させる
                emit('message deleted', {'id': msg_id}, broadcast=True)
    except Exception as e:
        print(f"削除失敗だッ！: {e}")
if __name__ == "__main__":
    # host="0.0.0.0" にすることで、Macの外（スマホ）からのアクセスを許可するぜッ！
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)

