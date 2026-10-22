from flask import Flask, render_template, Response, request
# フレームワーク（土台として利用するプログラムのセット）のFlaskから機能を呼び出す
from flask_socketio import SocketIO, emit
# ブラウザとサーバー間で双方向するための通信を設立
from datetime import datetime
# データを保存した時の日時を保存するもの
from pymongo import MongoClient
from bson.objectid import ObjectId
from urllib.parse import quote
# データベースに接続してデータを保存したり検索したり固有IDを扱ったりするもの
import certifi
# 安全に接続するための証明書
import base64
# 画像などのバイナリデータ(0と１の二進数で表現されたデータ)を文字コードに変換してやりとりするための仕組み
import gridfs
# DBに大きなファイルを保存するための仕組み
import traceback
# エラーが起きた時にどこで失敗したのかを追跡する
import cv2
# 画像処理を行う定番ライブラリのopenCVより画像の色の解析などを行う
import numpy as np
# 数値計算を高速で行うための必須ライブラリであるnumpy
import io
import csv
# CSVファイルをメモリ上で読み書きするための道具

# エクセル生成と自動グラフ化用のライブラリ
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.chart import ScatterChart, Series, Reference
# プログラムから綺麗なエクセルファイルを制作したりグラフを自動で書かせたりするための道具
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
# 機械学習の機能、ガウス過程回帰を読み出している

app = Flask(__name__)
# Flask（）でウェブアプリの仕組みを実体化している

socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=20000000)

# 「MongoDB接続設定」
mongo_uri = "mongodb+srv://e221239_db_user:Y0u667841you@cluster0.1tmhycc.mongodb.net/?appName=Cluster0"
client = MongoClient(mongo_uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=20000)
db = client["SNS"]
fs = gridfs.GridFS(db)

# =============================================================
# 🌐 画面遷移ルーティング設定
# =============================================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict")
def predict_page():
    return render_template("predict.html")

def get_image_data(image_id):
    try:
        if not fs.exists(image_id):
            return ""
        image_file = fs.get(image_id).read()
        return base64.b64encode(image_file).decode('utf-8')
    except Exception as e:
        print(f"Error: {e}")
        return ""

# =============================================================
# 📥 Socket.IO 通信イベント群
# =============================================================

@socketio.on('load messages')
def load_messages():
    try:
        images = db.images.find().sort('_id', -1).limit(50)
        spectrums = db.spectrums.find().sort('_id', -1).limit(50)
        
        combined_timeline = []
        
        for msg in images:
            img_b64 = get_image_data(msg['image_id'])
            combined_timeline.append({
                'id': str(msg['_id']),
                'date': msg.get('date', datetime.now()),
                'project_name': msg.get('project_name', '未分類'),
                'material_name': msg.get('material_name', '名称不明'),
                'message': msg.get('message', ''),
                'h_val': msg.get('h_val', 'N/A'),
                's_val': msg.get('s_val', 'N/A'),
                'v_val': msg.get('v_val', msg.get('avg_color', 'N/A')),
                'image_data': f"data:image/jpeg;base64,{img_b64}" if img_b64 else "",
                'type': 'image'
            })
            
        for spec in spectrums:
            x_vals = spec.get('x_values', spec.get('wavelengths', []))
            y_vals = spec.get('y_values', spec.get('intensities', []))
            x_lbl = spec.get('x_label', 'Wavelength (nm)')
            y_lbl = spec.get('y_label', 'Intensity')
            
            data_points = len(x_vals)
            combined_timeline.append({
                'id': str(spec['_id']),
                'date': spec.get('date', datetime.now()),
                'project_name': spec.get('project_name', '未分類'),
                'material_name': spec.get('material_name', '名称不明'),
                'message': f"【実験データ】{data_points} 点の {x_lbl} / {y_lbl} データを保持",
                'h_val': 'N/A', 's_val': 'N/A', 'v_val': 'N/A',
                'image_data': "",
                'type': 'csv',
                'x_label': x_lbl,
                'y_label': y_lbl,
                'x_values': x_vals,
                'y_values': y_vals,
                'wavelengths': x_vals,
                'intensities': y_vals
            })
            
        combined_timeline.sort(key=lambda x: x['date'])
        
        for item in combined_timeline:
            if isinstance(item['date'], datetime):
                item['date'] = item['date'].strftime('%Y-%m-%d %H:%M:%S')
        
        emit('load all messages', combined_timeline)
        
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

            nparray = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparray, cv2.IMREAD_COLOR)

            h_val, s_val, v_val = 0, 0, 0
            if img is not None:
                hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                avg_hsv = np.mean(hsv_img, axis=(0, 1))
                h_val = round(float(avg_hsv[0]), 1)
                s_val = round(float(avg_hsv[1]), 1)
                v_val = round(float(avg_hsv[2]), 1)

            image_id = fs.put(image_bytes, filename="upload.jpg")
            db.images.insert_one({
                'project_name': project_name,
                'material_name': material_name,
                'image_id': image_id,
                'message': message,
                'h_val': h_val,
                's_val': s_val,
                'v_val': v_val,
                'avg_color': v_val,
                'date': datetime.now()
            })

            load_messages()

    except Exception as e:
        traceback.print_exc()

@socketio.on('send csv data')
def handle_csv_data(data):
    try:
        project_name = data.get('project_name', '未分類')
        material_name = data.get('material_name', '名称不明')
        csv_text = data.get('csv_text', '')
        
        x_label = data.get('x_label', 'Wavelength (nm)')
        y_label = data.get('y_label', 'Intensity')

        x_values = []
        y_values = []

        f = io.StringIO(csv_text)
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            try:
                x_val = float(row[0].strip())
                y_val = float(row[1].strip())
                x_values.append(x_val)
                y_values.append(y_val)
            except ValueError:
                continue

        if x_values and y_values:
            result = db.spectrums.insert_one({
                'project_name': project_name,
                'material_name': material_name,
                'x_label': x_label,
                'y_label': y_label,
                'x_values': x_values,
                'y_values': y_values,
                'wavelengths': x_values,
                'intensities': y_values,
                'date': datetime.now()
            })
            
            emit('load one message', {
                'id': str(result.inserted_id),
                "project_name": project_name,
                "material_name": material_name,
                "message": f"【実験データ】{len(x_values)} 点の {x_label} / {y_label} データを保持",
                "h_val": "N/A", "s_val": "N/A", "v_val": "N/A",
                "image_data": "",
                "type": "csv",
                "x_label": x_label,
                "y_label": y_label,
                "x_values": x_values,
                "y_values": y_values,
                "wavelengths": x_values,
                "intensities": y_values
            }, broadcast=True)

    except Exception as e:
        traceback.print_exc()

@socketio.on('delete message')
def delete_message(data):
    try:
        msg_id = data.get('id')
        if not msg_id:
            return

        target_image = db.images.find_one({"_id": ObjectId(msg_id)})
        if target_image:
            fs.delete(target_image['image_id'])
            db.images.delete_one({"_id": ObjectId(msg_id)})
            emit('message deleted', {'id': msg_id}, broadcast=True)
            return

        target_spec = db.spectrums.find_one({"_id": ObjectId(msg_id)})
        if target_spec:
            db.spectrums.delete_one({"_id": ObjectId(msg_id)})
            emit('message deleted', {'id': msg_id}, broadcast=True)
            return

    except Exception as e:
        traceback.print_exc()

# =============================================================
# 💾 HTTPルート群（項目別ダウンロード＆エクセル自動グラフ化）
# =============================================================

@app.route("/download_csv")
def download_csv():
    try:
        target_project = request.args.get('project', 'すべて')
        query = {}
        if target_project != 'すべて':
            query['project_name'] = target_project
            
        spectrums = db.spectrums.find(query).sort('date', 1)
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        writer.writerow(['Date', 'Project', 'Material', 'Axis_X_Type', 'X_Value', 'Axis_Y_Type', 'Y_Value'])
        
        for spec in spectrums:
            dt = spec.get('date')
            date_str = dt.strftime('%Y-%m-%d %H:%M:%S') if dt else ''
            p_name = spec.get('project_name', '未分類')
            m_name = spec.get('material_name', '名称不明')
            x_lbl = spec.get('x_label', 'Wavelength (nm)')
            y_lbl = spec.get('y_label', 'Intensity')
            x_list = spec.get('x_values', spec.get('wavelengths', []))
            y_list = spec.get('y_values', spec.get('intensities', []))
            
            for x, y in zip(x_list, y_list):
                writer.writerow([date_str, p_name, m_name, x_lbl, x, y_lbl, y])
            
        output.seek(0)
        sjis_data = output.getvalue().encode('cp932', errors='replace')
        
        raw_filename = f"mi_data_report_{target_project}.csv" if target_project != 'すべて' else "mi_data_report_all.csv"
        encoded_filename = quote(raw_filename)
        
        return Response(
            sjis_data,
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename*=UTF-8''{encoded_filename}"}
        )
    except Exception as e:
        traceback.print_exc()
        return "CSV生成エラー", 500

@app.route("/download_excel")
def download_excel():
    try:
        target_project = request.args.get('project', 'すべて')
        query = {}
        if target_project != 'すべて':
            query['project_name'] = target_project

        wb = Workbook()
        
        ws1 = wb.active
        ws1.title = "RawPlotData"
        
        spectrums = list(db.spectrums.find(query).sort('date', 1))
        
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        ws1.append(['Project', 'Material', 'X_Label', 'X_Value', 'Y_Label', 'Y_Value'])
        for cell in ws1[1]:
            cell.fill = PatternFill(start_color="5B9BD5", end_color="5B9BD5", fill_type="solid")
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")
            
        row_count1 = 1
        series_list = []
        
        main_title = "Material Analysis Chart"
        if spectrums:
            main_title = f"{spectrums[0].get('y_label','Y')} vs {spectrums[0].get('x_label','X')}"

        for spec in spectrums:
            p_name = spec.get('project_name', '未分類')
            m_name = spec.get('material_name', '名称不明')
            x_lbl = spec.get('x_label', 'X')
            y_lbl = spec.get('y_label', 'Y')
            x_list = spec.get('x_values', spec.get('wavelengths', []))
            y_list = spec.get('y_values', spec.get('intensities', []))
            
            start_row = row_count1 + 1
            for x, y in zip(x_list, y_list):
                ws1.append([p_name, m_name, x_lbl, x, y_lbl, y])
                row_count1 += 1
            end_row = row_count1
            
            if start_row <= end_row:
                xvalues = Reference(ws1, min_col=4, min_row=start_row, max_row=end_row)
                yvalues = Reference(ws1, min_col=6, min_row=start_row, max_row=end_row)
                ser = Series(yvalues, xvalues=xvalues, title=f"{m_name} ({p_name})")
                
                ser.graphicalProperties.line.width = 25000
                ser.marker.symbol = "none"
                series_list.append(ser)

        for col in ws1.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = chr(64 + col[0].column)
            ws1.column_dimensions[col_letter].width = max(max_len + 3, 12)

        if series_list:
            chart = ScatterChart()
            chart.title = f"{main_title} ({target_project})"
            chart.style = 13
            chart.x_axis.title = spectrums[0].get('x_label','X軸') if spectrums else "X Axis"
            chart.y_axis.title = spectrums[0].get('y_label','Y軸') if spectrums else "Y Axis"
            chart.scatterStyle = "line"
            for ser in series_list:
                chart.series.append(ser)
            ws1.add_chart(chart, "H2")

        ws2 = wb.create_sheet(title="ImageAnalysisLog")
        ws2.append(['Date', 'Project', 'Material', 'Hue (色相)', 'Saturation (彩度)', 'Value (明度)', 'Message'])
        for cell in ws2[1]:
            cell.fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")
            
        messages = db.images.find(query).sort('date', 1)
        for msg in messages:
            dt = msg.get('date')
            date_str = dt.strftime('%Y-%m-%d %H:%M:%S') if dt else ''
            ws2.append([
                date_str,
                msg.get('project_name', '未分類'),
                msg.get('material_name', '名称不明'),
                msg.get('h_val', 'N/A'),
                msg.get('s_val', 'N/A'),
                msg.get('v_val', msg.get('avg_color', 'N/A')),
                msg.get('message', '')
            ])

        for col in ws2.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = chr(64 + col[0].column)
            ws2.column_dimensions[col_letter].width = max(max_len + 3, 12)

        output = io.BytesIO()
        wb.save(output)
        output.seek(0)
        
        raw_filename = f"universal_mi_report_{target_project}.xlsx" if target_project != 'すべて' else "universal_mi_report.xlsx"
        encoded_filename = quote(raw_filename)

        return Response(
            output.getvalue(),
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-disposition": f"attachment; filename*=UTF-8''{encoded_filename}"}
        )
    except Exception as e:
        traceback.print_exc()
        return "Excel生成エラー", 500

@app.route("/api/predict_dynamic", methods=["POST"])
def predict_dynamic():
    try:
        req_data = request.get_json() or {}
        target_temp = float(req_data.get("temperature", 25.0))
        target_mat_a = float(req_data.get("material_a", 1.5))
        target_mat_b = float(req_data.get("material_b", 0.5))

        latest_spec = db.spectrums.find_one(sort=[('_id', -1)])
        if not latest_spec:
            return {"error": "データベースに学習元のCSVデータがありません。"}, 400

        x_vals = latest_spec.get('x_values', [])
        y_vals = latest_spec.get('y_values', [])

        if len(x_vals) < 5:
            return {"error": "学習させるためのデータ点が足りません(最低5点以上必要です)。"}, 400

        X_train = []
        y_train = []
        for t, val in zip(x_vals, y_vals):
            X_train.append([t, 25.0, 1.5, 0.5]) 
            y_train.append(val)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0, 10.0, 1.0, 1.0], (1e-2, 1e2)) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.0)

        gp.fit(X_train, y_train)

        max_time = max(x_vals)
        predict_times = np.linspace(0, max_time, 100)
        
        X_predict = []
        for pt in predict_times:
            X_predict.append([pt, target_temp, target_mat_a, target_mat_b])
        X_predict = np.array(X_predict)

        y_pred, y_std = gp.predict(X_predict, return_std=True)

        response_data = {
            "times": predict_times.tolist(),
            "predictions": y_pred.tolist(),
            "confidence_high": (y_pred + 1.96 * y_std).tolist(),
            "confidence_low": (y_pred - 1.96 * y_std).tolist(),
            "target_conditions": {
                "temperature": target_temp,
                "material_a": target_mat_a,
                "material_b": target_mat_b
            }
        }
        return response_data

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}, 500

if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)