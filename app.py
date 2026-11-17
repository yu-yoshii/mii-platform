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
import os
import tempfile
from pathlib import Path
import json
# CSVファイルをメモリ上で読み書きするための道具

# エクセル生成と自動グラフ化用のライブラリ
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.chart import ScatterChart, Series, Reference
# プログラムから綺麗なエクセルファイルを制作したりグラフを自動で書かせたりするための道具
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
# 機械学習の機能、ガウス過程回帰を読み出している

app = Flask(__name__)
# Flask（）でウェブアプリの仕組みを実体化している
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
MAX_GPR_TRAINING_POINTS = 1000
DEFAULT_GPR_TRAINING_POINTS = 250

socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=20000000)

# 「MongoDB接続設定」
mongo_uri = "mongodb+srv://e221239_db_user:Y0u667841you@cluster0.1tmhycc.mongodb.net/?appName=Cluster0"
client = MongoClient(mongo_uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=20000)
db = client["SNS"]
fs = gridfs.GridFS(db)

# =============================================================
#  画面遷移ルーティング設定
# =============================================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict")
def predict_page():
    return render_template(
        "predict.html",
        max_training_points=MAX_GPR_TRAINING_POINTS,
        default_training_points=DEFAULT_GPR_TRAINING_POINTS
    )

def experiment_csv_path(experiment_id):
    return Path(app.instance_path) / "experiments" / f"{experiment_id}.csv"

def save_experiment_csv(experiment_id, x_values, y_values):
    path = experiment_csv_path(experiment_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            delete=False
        ) as output:
            temporary_path = Path(output.name)
            writer = csv.writer(output)
            writer.writerow(["x_value", "y_value"])
            writer.writerows(zip(x_values, y_values))

        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return path

def read_experiment_csv(path):
    x_values = []
    y_values = []

    with path.open("r", newline="", encoding="utf-8") as source:
        for row in csv.reader(source):
            if len(row) < 2:
                continue
            try:
                x_value = float(row[0].strip())
                y_value = float(row[1].strip())
            except ValueError:
                continue
            if np.isfinite(x_value) and np.isfinite(y_value):
                x_values.append(x_value)
                y_values.append(y_value)

    return x_values, y_values

def ensure_experiment_csv(spec):
    path = experiment_csv_path(spec["_id"])
    if not path.is_file():
        x_values = spec.get("x_values", spec.get("wavelengths", []))
        y_values = spec.get("y_values", spec.get("intensities", []))
        if not x_values or not y_values:
            raise ValueError("実験データに学習可能な値がありません。")
        save_experiment_csv(spec["_id"], x_values, y_values)
    return path

def select_evenly_spaced_points(x_values, y_values, point_count):
    indices = np.linspace(0, len(x_values) - 1, num=point_count, dtype=int)
    return np.asarray(x_values)[indices], np.asarray(y_values)[indices]

def parse_csv_table(csv_bytes):
    try:
        csv_text = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        csv_text = csv_bytes.decode("cp932")

    table = list(csv.reader(io.StringIO(csv_text)))
    if not table:
        raise ValueError("CSVにデータがありません。")

    headers = [header.strip() for header in table[0]]
    if len(headers) < 2 or any(not header for header in headers):
        raise ValueError("CSVの1行目に空欄のない列名を2つ以上設定してください。")
    if len(set(headers)) != len(headers):
        raise ValueError("CSVの列名が重複しています。列名をそれぞれ異なるものにしてください。")

    rows = []
    for row_number, row in enumerate(table[1:], start=2):
        if not any(value.strip() for value in row):
            continue
        if len(row) != len(headers):
            raise ValueError(f"{row_number}行目の列数が見出しと一致しません。")
        rows.append([value.strip() for value in row])

    return headers, rows

def parse_numeric_columns(headers, rows, input_columns, target_columns):
    if not isinstance(input_columns, list) or not isinstance(target_columns, list):
        raise ValueError("条件列と結果列を選択してください。")
    if not input_columns or not target_columns:
        raise ValueError("条件列と結果列をそれぞれ1列以上選択してください。")
    if any(not isinstance(column, str) for column in input_columns + target_columns):
        raise ValueError("列名の指定が正しくありません。")
    if len(set(input_columns)) != len(input_columns) or len(set(target_columns)) != len(target_columns):
        raise ValueError("同じ列を重複して選択できません。")
    if set(input_columns) & set(target_columns):
        raise ValueError("同じ列を条件列と結果列の両方には指定できません。")
    if any(column not in headers for column in input_columns + target_columns):
        raise ValueError("選択した列がCSVにありません。CSVを選び直してください。")

    selected_columns = input_columns + target_columns
    column_indices = [headers.index(column) for column in selected_columns]
    numeric_rows = []
    for row_number, row in enumerate(rows, start=2):
        try:
            values = [float(row[index]) for index in column_indices]
        except ValueError as error:
            raise ValueError(f"{row_number}行目に数値でない値があります。") from error
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{row_number}行目に無限大または無効な数値があります。")
        numeric_rows.append(values)

    if len(numeric_rows) < 5:
        raise ValueError("機械学習には5行以上の測定データが必要です。")

    values = np.asarray(numeric_rows, dtype=float)
    return values[:, :len(input_columns)], values[:, len(input_columns):]

def select_evenly_spaced_rows(input_values, target_values, row_count):
    indices = np.linspace(0, len(input_values) - 1, num=row_count, dtype=int)
    return np.asarray(input_values)[indices], np.asarray(target_values)[indices]

def get_experiment_model(spec, input_columns, target_columns, input_values, target_values, requested_points):
    cache_key = (
        str(spec["_id"]),
        tuple(input_columns),
        tuple(target_columns),
        requested_points
    )
    if model_cache["cache_key"] != cache_key or model_cache.get("gp_model") is None:
        print("🔄 実験データから多次元・複数結果のGPRモデルを学習します...")
        x_train, y_train = select_evenly_spaced_rows(
            input_values, target_values, requested_points
        )
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(x_train)
        kernel = (
            C(1.0, (1e-3, 1e3))
            * RBF(np.ones(len(input_columns)), (1e-2, 1e2))
            + WhiteKernel(noise_level=0.1)
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=1,
            alpha=1e-6,
            normalize_y=True
        )
        gp.fit(scaled_train, y_train)
        model_cache["cache_key"] = cache_key
        model_cache["scaler"] = scaler
        model_cache["gp_model"] = gp
    else:
        print("⚡ キャッシュされたGPRモデルを使用します")

    return model_cache["scaler"], model_cache["gp_model"]

def load_experiment_table(spec):
    input_columns = spec.get("input_columns")
    target_columns = spec.get("target_columns")
    cache_key = (
        str(spec["_id"]),
        str(spec.get("csv_file_id", "legacy")),
        tuple(input_columns or []),
        tuple(target_columns or []),
        spec.get("x_label"),
        spec.get("y_label")
    )
    if experiment_table_cache["cache_key"] == cache_key:
        return experiment_table_cache["table"]

    if input_columns and target_columns and spec.get("csv_file_id"):
        csv_bytes = fs.get(spec["csv_file_id"]).read()
        headers, rows = parse_csv_table(csv_bytes)
        input_values, target_values = parse_numeric_columns(
            headers, rows, input_columns, target_columns
        )
        table = (input_columns, target_columns, input_values, target_values)
    else:
        x_values, y_values = read_experiment_csv(ensure_experiment_csv(spec))
        input_name = spec.get("x_label", "X")
        target_name = spec.get("y_label", "Y")
        table = (
            [input_name],
            [target_name],
            np.asarray(x_values, dtype=float).reshape(-1, 1),
            np.asarray(y_values, dtype=float).reshape(-1, 1)
        )

    experiment_table_cache["cache_key"] = cache_key
    experiment_table_cache["table"] = table
    return table

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
#  Socket.IO 通信イベント群
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
            input_columns = spec.get("input_columns")
            target_columns = spec.get("target_columns")
            if input_columns and target_columns:
                x_vals = []
                y_vals = []
                x_lbl = ", ".join(input_columns)
                y_lbl = ", ".join(target_columns)
                data_points = spec.get("row_count", 0)
                message = (
                    f"【{spec.get('domain', '実験データ')}】{data_points} 行"
                    f" / 条件: {x_lbl} / 結果: {y_lbl}"
                )
            else:
                x_vals = spec.get('x_values', spec.get('wavelengths', []))
                y_vals = spec.get('y_values', spec.get('intensities', []))
                x_lbl = spec.get('x_label', 'Wavelength (nm)')
                y_lbl = spec.get('y_label', 'Intensity')
                data_points = len(x_vals)
                message = f"【実験データ】{data_points} 点の {x_lbl} / {y_lbl} データを保持"

            combined_timeline.append({
                'id': str(spec['_id']),
                'date': spec.get('date', datetime.now()),
                'project_name': spec.get('project_name', '未分類'),
                'material_name': spec.get('material_name', '名称不明'),
                'domain': spec.get('domain', '未分類'),
                'message': message,
                'h_val': 'N/A', 's_val': 'N/A', 'v_val': 'N/A',
                'image_data': "",
                'type': 'csv',
                'row_count': data_points,
                'input_columns': input_columns or [x_lbl],
                'target_columns': target_columns or [y_lbl],
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

@app.route("/api/experiments")
def list_experiments():
    try:
        experiments = []
        for spec in db.spectrums.find().sort("date", -1):
            input_columns = spec.get("input_columns")
            target_columns = spec.get("target_columns")
            if input_columns and target_columns and spec.get("csv_file_id"):
                point_count = spec.get("row_count", 0)
            else:
                input_columns, target_columns, input_values, target_values = load_experiment_table(spec)
                point_count = min(len(input_values), len(target_values))
            date = spec.get("date")
            experiments.append({
                "id": str(spec["_id"]),
                "project_name": spec.get("project_name", "未分類"),
                "material_name": spec.get("material_name", "名称不明"),
                "domain": spec.get("domain", "未分類"),
                "date": date.isoformat() if isinstance(date, datetime) else str(date or ""),
                "input_columns": input_columns,
                "target_columns": target_columns,
                "point_count": point_count
            })
        return {"experiments": experiments}
    except Exception:
        traceback.print_exc()
        return {"error": "実験データ一覧の読み込みに失敗しました。"}, 500

@app.route("/api/csv/preview", methods=["POST"])
def preview_csv():
    uploaded_file = request.files.get("file")
    if not uploaded_file or not uploaded_file.filename:
        return {"error": "CSVファイルを選択してください。"}, 400

    try:
        headers, rows = parse_csv_table(uploaded_file.read())
        return {"columns": headers, "preview": rows[:5], "row_count": len(rows)}
    except (UnicodeDecodeError, csv.Error, ValueError) as error:
        return {"error": str(error)}, 400

@app.route("/api/experiments", methods=["POST"])
def create_experiment():
    uploaded_file = request.files.get("file")
    if not uploaded_file or not uploaded_file.filename:
        return {"error": "CSVファイルを選択してください。"}, 400

    try:
        csv_bytes = uploaded_file.read()
        headers, rows = parse_csv_table(csv_bytes)
        input_columns = json.loads(request.form.get("input_columns", "[]"))
        target_columns = json.loads(request.form.get("target_columns", "[]"))
        input_values, target_values = parse_numeric_columns(
            headers, rows, input_columns, target_columns
        )

        experiment_id = ObjectId()
        file_id = fs.put(
            csv_bytes,
            filename=uploaded_file.filename,
            content_type="text/csv",
            metadata={"experiment_id": str(experiment_id), "kind": "experiment_data"}
        )
        try:
            db.spectrums.insert_one({
                "_id": experiment_id,
                "project_name": request.form.get("project_name", "未分類"),
                "material_name": request.form.get("material_name", "名称不明"),
                "domain": request.form.get("domain", "未分類"),
                "csv_filename": uploaded_file.filename,
                "csv_file_id": file_id,
                "input_columns": input_columns,
                "target_columns": target_columns,
                "row_count": len(input_values),
                "date": datetime.now()
            })
        except Exception:
            fs.delete(file_id)
            raise

        return {
            "id": str(experiment_id),
            "input_columns": input_columns,
            "target_columns": target_columns,
            "row_count": len(input_values)
        }, 201
    except (UnicodeDecodeError, csv.Error, ValueError, json.JSONDecodeError) as error:
        return {"error": str(error)}, 400
    except Exception:
        traceback.print_exc()
        return {"error": "実験データの保存に失敗しました。"}, 500

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
            experiment_id = ObjectId()
            file_path = save_experiment_csv(experiment_id, x_values, y_values)
            try:
                result = db.spectrums.insert_one({
                    '_id': experiment_id,
                    'project_name': project_name,
                    'material_name': material_name,
                    'csv_filename': file_path.name,
                    'x_label': x_label,
                    'y_label': y_label,
                    'x_values': x_values,
                    'y_values': y_values,
                    'wavelengths': x_values,
                    'intensities': y_values,
                    'date': datetime.now()
                })
            except Exception:
                file_path.unlink(missing_ok=True)
                raise
            
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
            if target_spec.get("csv_file_id"):
                fs.delete(target_spec["csv_file_id"])
            else:
                experiment_csv_path(target_spec["_id"]).unlink(missing_ok=True)
            emit('message deleted', {'id': msg_id}, broadcast=True)
            return

    except Exception as e:
        traceback.print_exc()

# =============================================================
#  HTTPルート群（項目別ダウンロード＆エクセル自動グラフ化）
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
#  サーバーのメモリ上に学習済みAIモデルを保持するためのキャッシュ箱
experiment_table_cache = {
    "cache_key": None,
    "table": None
}
model_cache = {
    "cache_key": None,
    "scaler": None,
    "gp_model": None
}

@app.route("/api/predict_dynamic", methods=["POST"])
def predict_dynamic():
    try:
        req_data = request.get_json() or {}
        experiment_id = req_data.get("experiment_id")
        if not isinstance(experiment_id, str) or not ObjectId.is_valid(experiment_id):
            return {"error": "学習対象の実験を選択してください。"}, 400

        spec = db.spectrums.find_one({"_id": ObjectId(experiment_id)})
        if not spec:
            return {"error": "選択した実験データが見つかりません。"}, 404

        input_columns, target_columns, input_values, target_values = load_experiment_table(spec)
        point_total = len(input_values)
        if point_total < 5:
            return {"error": "機械学習には5行以上の測定データが必要です。"}, 400

        requested_points = req_data.get(
            "training_points",
            min(DEFAULT_GPR_TRAINING_POINTS, point_total)
        )
        if isinstance(requested_points, bool) or not isinstance(requested_points, int):
            return {"error": "学習点数は整数で指定してください。"}, 400
        max_training_points = min(point_total, MAX_GPR_TRAINING_POINTS)
        if requested_points < 5 or requested_points > max_training_points:
            return {
                "error": (
                    f"学習点数は5〜{max_training_points}の範囲で指定してください。"
                    f"登録データ全{point_total}行から均等に抽出します。"
                )
            }, 400

        conditions = req_data.get("conditions")
        if not isinstance(conditions, dict) or set(conditions) != set(input_columns):
            return {"error": "すべての実験条件を入力してください。"}, 400

        try:
            query_values = [float(conditions[column]) for column in input_columns]
        except (TypeError, ValueError):
            return {"error": "実験条件は数値で入力してください。"}, 400
        if not np.all(np.isfinite(query_values)):
            return {"error": "実験条件に無効な数値があります。"}, 400

        scaler, gp_model = get_experiment_model(
            spec, input_columns, target_columns,
            input_values, target_values, requested_points
        )
        query = np.asarray([query_values], dtype=float)
        y_pred, y_std = gp_model.predict(scaler.transform(query), return_std=True)
        prediction_values = np.asarray(y_pred).reshape(-1)
        std_values = np.asarray(y_std).reshape(-1)

        return {
            "experiment_id": str(spec["_id"]),
            "training_points": requested_points,
            "total_points": point_total,
            "input_conditions": dict(zip(input_columns, query_values)),
            "predictions": dict(zip(target_columns, prediction_values.tolist())),
            "confidence_high": dict(zip(
                target_columns,
                (prediction_values + 1.96 * std_values).tolist()
            )),
            "confidence_low": dict(zip(
                target_columns,
                (prediction_values - 1.96 * std_values).tolist()
            ))
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}, 500

@app.route("/api/predict_grid", methods=["POST"])
def predict_grid():
    try:
        req_data = request.get_json() or {}
        experiment_id = req_data.get("experiment_id")
        if not isinstance(experiment_id, str) or not ObjectId.is_valid(experiment_id):
            return {"error": "学習対象の実験を選択してください。"}, 400

        spec = db.spectrums.find_one({"_id": ObjectId(experiment_id)})
        if not spec:
            return {"error": "選択した実験データが見つかりません。"}, 404

        input_columns, target_columns, input_values, target_values = load_experiment_table(spec)
        point_total = len(input_values)
        max_training_points = min(point_total, MAX_GPR_TRAINING_POINTS)
        requested_points = req_data.get(
            "training_points",
            min(DEFAULT_GPR_TRAINING_POINTS, point_total)
        )
        if (
            isinstance(requested_points, bool)
            or not isinstance(requested_points, int)
            or requested_points < 5
            or requested_points > max_training_points
        ):
            return {
                "error": f"学習点数は5〜{max_training_points}の範囲で指定してください。"
            }, 400

        conditions = req_data.get("conditions")
        if not isinstance(conditions, dict) or set(conditions) != set(input_columns):
            return {"error": "すべての実験条件を入力してください。"}, 400
        try:
            fixed_values = np.asarray(
                [float(conditions[column]) for column in input_columns],
                dtype=float
            )
        except (TypeError, ValueError):
            return {"error": "実験条件は数値で入力してください。"}, 400
        if not np.all(np.isfinite(fixed_values)):
            return {"error": "実験条件に無効な数値があります。"}, 400

        x_column = req_data.get("x_column")
        y_column = req_data.get("y_column")
        if x_column not in input_columns:
            return {"error": "グラフの横軸を選び直してください。"}, 400
        if y_column is not None and (
            y_column not in input_columns or y_column == x_column
        ):
            return {"error": "グラフの縦軸を選び直してください。"}, 400

        grid_size = req_data.get("grid_size", 40)
        max_grid_size = 100 if y_column is None else 60
        if (
            isinstance(grid_size, bool)
            or not isinstance(grid_size, int)
            or grid_size < 5
            or grid_size > max_grid_size
        ):
            return {
                "error": f"グラフの細かさは5〜{max_grid_size}の範囲で指定してください。"
            }, 400

        x_index = input_columns.index(x_column)
        x_values = np.linspace(
            float(np.min(input_values[:, x_index])),
            float(np.max(input_values[:, x_index])),
            grid_size
        )
        if y_column is None:
            queries = np.tile(fixed_values, (grid_size, 1))
            queries[:, x_index] = x_values
            y_values = None
        else:
            y_index = input_columns.index(y_column)
            y_values = np.linspace(
                float(np.min(input_values[:, y_index])),
                float(np.max(input_values[:, y_index])),
                grid_size
            )
            x_mesh, y_mesh = np.meshgrid(x_values, y_values)
            queries = np.tile(fixed_values, (grid_size * grid_size, 1))
            queries[:, x_index] = x_mesh.ravel()
            queries[:, y_index] = y_mesh.ravel()

        scaler, gp_model = get_experiment_model(
            spec, input_columns, target_columns,
            input_values, target_values, requested_points
        )
        predicted, predicted_std = gp_model.predict(
            scaler.transform(queries),
            return_std=True
        )
        prediction_values = np.asarray(predicted)
        std_values = np.asarray(predicted_std)

        if prediction_values.ndim == 1:
            prediction_values = prediction_values.reshape(-1, 1)
            std_values = std_values.reshape(-1, 1)
        if y_values is not None:
            prediction_values = prediction_values.reshape(
                grid_size, grid_size, len(target_columns)
            )
            std_values = std_values.reshape(
                grid_size, grid_size, len(target_columns)
            )

        response = {
            "x_column": x_column,
            "y_column": y_column,
            "x_values": x_values.tolist(),
            "y_values": y_values.tolist() if y_values is not None else None,
            "training_points": requested_points,
            "total_points": point_total,
            "fixed_conditions": {
                column: float(fixed_values[index])
                for index, column in enumerate(input_columns)
                if column not in (x_column, y_column)
            }
        }
        for target_index, target_column in enumerate(target_columns):
            mean_values = prediction_values[..., target_index]
            confidence_values = std_values[..., target_index]
            response.setdefault("predictions", {})[target_column] = mean_values.tolist()
            response.setdefault("confidence_low", {})[target_column] = (
                mean_values - 1.96 * confidence_values
            ).tolist()
            response.setdefault("confidence_high", {})[target_column] = (
                mean_values + 1.96 * confidence_values
            ).tolist()
        return response
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}, 500

if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)