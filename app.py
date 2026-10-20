from flask import Flask, render_template, Response, request
#フレームワーク（土台として利用するプログラムのセット）のFlaskから機能を呼び出す
from flask_socketio import SocketIO, emit
#ブラウザとサーバー間で双方向するための通信を設立
from datetime import datetime
#データを保存した時の日時を保存するもの
from pymongo import MongoClient
from bson.objectid import ObjectId
from urllib.parse import quote
#データベースに接続してデータを保存したり検索したり固有IDを扱ったりするもの
import certifi
#安全に接続するための証明書
import base64
# 画像などのバイナリデータ(0と１の二進数で表現されたデータ)を文字コードに変換してやりとりするための仕組み
import gridfs
# DBに大きなファイルを保存するための仕組み
import traceback
# エラーが起きた時にどこで失敗したのかを追跡する
import cv2
# 画像処理を行う定番ライブラリのopenCVより画像の色の解析などを行う
import numpy as np
#数値計算を高速で行うための必須ライブラリであるnumpy
import io
import csv
#CSVファイルをメモリ上で読み書きするための道具

# エクセル生成と自動グラフ化用のライブラリ
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.chart import ScatterChart, Series, Reference
# プログラムから綺麗なエクセルファイルを制作したりグラフを自動で書かせたりするための道具
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
#機械学習の機能、ガウス過程回帰を読み出している

app = Flask(__name__)
# Flask（）でウェブアプリの仕組みを実体化している
# （__name__)で今動かしているファイルの名前をFlaskに知らせるお決まりの書き方。これによりこのファイルがウェブアプリの中心であるとPythonに認識させappという箱にしまっている



socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=20000000)
# 先ほど作ったウェブアプリの本体であるappをベースにしてリアルタイム通信の機能を持った新しいしれ一応を作っている
# 2個目はどのWebページからのアクセスも許可するというもの
# 3個目は一度に遅れるデータの最大容量を大きめに設定している
# 「MongoDB接続設定」
mongo_uri = "mongodb+srv://e221239_db_user:Y0u667841you@cluster0.1tmhycc.mongodb.net/?appName=Cluster0"
# DBにログインするための接続アドレスを文字列として保存
client = MongoClient(mongo_uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=20000)
#先ほどのmogo_uriを使用してインターネット越しにデータベースのbサーバーへ接続を試みている
# 2個目は安全な通史のための証明書bを使っていねという意味
# 3個目は20秒繋がらなかったらタイムアウトしてという指示
db = client["SNS"]
#捨属したDBの中からSNSというDBを選択してdbという箱に入れている
fs = gridfs.GridFS(db)
# SNSDBの中に画像などの重たいデータを保存するためのファイル収容棚(Gridfs)を作りfsという名前でセットしている

@app.route("/")
def index():
    return render_template("index.html")

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
# @から始まる行はデコレータと呼ばれ下の関数に特別な役割も持たせる
# ウェブアプリ側からメッセージを読み込んでという合図が来た際に下の関数を自動で動かしてくれる
def load_messages():
    # defはここから新しい関数を作るという宣言
    # この関数の名前とデータを受け取るかを選択する
    try:
        #このなっかの関すはもしかしたらネットワークのエラーやデータの間違いで失敗するかもしれない危ない処理を囲むエリア
        #たとえ途中でバグやエラーが起きても強制終了してしまわないようにエラーをexcept Exception as eでキャッチする。
        #traceback.print_exc()でどこでエラーが起きたかをターミナルに詳しく表示させることが可能
        images = db.images.find().sort('_id', -1).limit(50)
        #プロフラミングにおいて.指示である。左から順に実行していく
        #db.images.find()でDBの中のimagesというコレクションから画像やメッセージのデータを全て探してもっと来ることが可能
        #.sort('_id',-1)でデータの並び順を整える-1はあたらしいじゅん（降順）を意味する
        #limit(50)はデータが何千件とあっても重くならないように最新の５０件だけに絞り込んでいる
        spectrums = db.spectrums.find().sort('_id', -1).limit(50)
        #spectrumsは同様にアップロードされたcsvの実験データを新しい順に50件引っ張ってきます
        
        combined_timeline = []
        #中身が空っぽのリストを新しく用意している
        
        for msg in images:
            #データベースから見つけてきた画像データを一軒ずつmsgという変数に取り出しながら下を繰り返せという意味する
            img_b64 = get_image_data(msg['image_id'])
            #DBには画像のIDだけが入っているので専用の関数get_image_dataを使って実際の画像ファイルを文字データ(base64方式)に変換して取り出している
            combined_timeline.append({
                #appendはリストの最後に新しい要素を追加する
                #()の中の{}(辞書型)を使用してブラウザがキーと値をペアにしている
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
                #msg.get('project_name',未分類)はもしデータベースにもしproject_nameがカラダとしても未分類という文字を入れろという指示
            })
            
        for spec in spectrums:
            # 💡 互換性担保のハック：新しい 'x_values' が無ければ過去の 'wavelengths' を、無ければ空配列を採用
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
                'message': f"【実験データ】{data_points} 点の {x_lbl} - {y_lbl} データを保持",
                'h_val': 'N/A', 's_val': 'N/A', 'v_val': 'N/A',
                'image_data': "",
                'type': 'csv',
                # フロント側へ新旧どちらのプロパティ名が来てもバグらないように両方持たせて投げる
                'x_label': x_lbl,
                'y_label': y_lbl,
                'x_values': x_vals,
                'y_values': y_vals,
                'wavelengths': x_vals,
                'intensities': y_vals
            })
            
        combined_timeline.sort(key=lambda x: x['date'])
        #.sort()は並び替える命令dateを見て古い順に並べる
        
        for item in combined_timeline:
            if isinstance(item['date'], datetime):
                item['date'] = item['date'].strftime('%Y-%m-%d %H:%M:%S')
        
        emit('load all messages', combined_timeline)
        #emit()はブラウザ側にデータを送るための命令
        
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

            # 画像追加時はタイムライン全体を再取得させるシグナルを飛ばすのが最も安全
            load_messages()

    except Exception as e:
        traceback.print_exc()

@socketio.on('send csv data')
def handle_csv_data(data):
    try:
        project_name = data.get('project_name', '未分類')
        material_name = data.get('material_name', '名称不明')
        csv_text = data.get('csv_text', '')
        
        # 画面から送ってきた動的なラベルを取得（なければ従来のデフォルトをハメる）
        x_label = data.get('x_label', 'Wavelength (nm)')
        y_label = data.get('y_label', 'Intensity')

        x_values = []
        y_values = []
        #あとで数値を入れるための空っぽなリストをそれぞれ用意する

        f = io.StringIO(csv_text)
        #画面から送られていたのは文字列であるためPythonにファイルのようなものとして扱わせるために仮想のファイルに変換する道具としてio.StringIOに入れている
        reader = csv.reader(f)
        #それをインポートしておいたcsvライブラリに渡すことでただの文字列だったものをcsvとして１行ずつ読めるように分解してくれる
        for row in reader:
            #csvの表を１行ずつ最後までループで読み込む
            if not row or len(row) < 2:
                #データが足りなくてえら０になりならないように次へ行くようにしている
                continue
            try:
                x_val = float(row[0].strip())
                y_val = float(row[1].strip())
                #csvの１列目の文字を取り出し前後の余計な空白を.strip()で消したあと計算できるようにfloat（小数点の値）に変換している楡爪も同じように数値にする
                x_values.append(x_val)
                y_values.append(y_val)
            except ValueError:
                continue
            #もしcsvの１行目が文字であってもエラーにならないように対策している

        if x_values and y_values:
            result = db.spectrums.insert_one({
                #集めたすべてのデータをDBに保存している
                'project_name': project_name,
                'material_name': material_name,
                'x_label': x_label,
                'y_label': y_label,
                'x_values': x_values,
                'y_values': y_values,
                # 過去データとの互換構造も維持してDBへ格納
                'wavelengths': x_values,
                'intensities': y_values,
                'date': datetime.now()
            })
            
            # 💡 新しいキー名とカスタムラベルを速報エミットにブレンド！
            emit('load one message', {
                # 保存が成功したらこのアプリを開いている全員に対してリアルタイムで飛ばす
                'id': str(result.inserted_id),
                "project_name": project_name,
                "material_name": material_name,
                "message": f"【実験データ】{len(x_values)} 点の {x_label} - {y_label} データを保持",
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
            #データベースから見つけてきたたくさんの実験データであるspectrumsから一軒ずつ取り出して　specという箱に入れる
            #その中からプロジェクト名、材料名、x、y軸のラベル、実際の数値のリストを安全に取り出してそれぞれわかりやすい名前の箱にしまっている
            
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
#ブラウザで/download_excelというアドレスが開かれた時に動かす
def download_excel():
    try:
        target_project = request.args.get('project', 'すべて')
        #URLの後ろについているぱらめーたからどのプロジェクトのデータをエクセルにまとめるかを受け取っている
        query = {}
        if target_project != 'すべて':
            query['project_name'] = target_project
            #データベースから条件に合うものだけを絞り込む準備をしている

        wb = Workbook()
        #インポートしておいたopenxlの機能を使用してまっさらなエクセルファイルを作成している
        
        ws1 = wb.active
        #一枚目のシート(ws1)に下のプログラムで名前を変更している
        ws1.title = "RawPlotData"
        
        spectrums = list(db.spectrums.find(query).sort('date', 1))
        #条件に合う実験データをかき集めている
        
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        ws1.append(['Project', 'Material', 'X_Label', 'X_Value', 'Y_Label', 'Y_Value'])
        for cell in ws1[1]:
            cell.fill = PatternFill(start_color="5B9BD5", end_color="5B9BD5", fill_type="solid")
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")
            
        row_count1 = 1
        series_list = []
        
        # 代表的な軸名を決める（グラフタイトル用）
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
            #あとでエクセルの中にグラフを書く時にこのデータの塊はエクセルの何業目から何行目に入っているかを正確に教える必要があるため今から描き始めるのは何行目かをメモしている
            for x, y in zip(x_list, y_list):
                #zip()はx軸とy軸のリストをまるで一つずつセットにしてまとめる命令
                ws1.append([p_name, m_name, x_lbl, x, y_lbl, y])
                #エクセルのシートに対して１行分のデータを一番下の行に追加している
                row_count1 += 1
                #１行書き込むごとに現在の行のカウントを+１増やしているこれによって何行書き込んだかがメモされる
            end_row = row_count1
            #終わった瞬間の行番号をend_rowとして保存する
            
            if start_row <= end_row:
                # 💡 改造：動的カラム位置（Col 4 がXの値、Col 6 がYの値）からリファレンスを生成
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
            #もしえくするに書き込むべきデータが一点以上存在しているばという条件で実行している。一つもデータがない時エラーが起きることを防いでいる
            chart = ScatterChart()
            #openpyxlの機能を使用して散布図の不￥グラフの枠組みを新しく作っている
            chart.title = f"{main_title} ({target_project})"
            chart.style = 13
            #13版のテンプレートを使用
            chart.x_axis.title = spectrums[0].get('x_label','X軸') if spectrums else "X Axis"
            chart.y_axis.title = spectrums[0].get('y_label','Y軸') if spectrums else "Y Axis"
            #縦軸横軸のラベルの文字をセットしている
            chart.scatterStyle = "line"
            #線で繋げという命令
            for ser in series_list:
                #先ほどのエクセルでも書き込みの時に準備していたグラフの線のデータ(series_list)を一本ずつにどんどん追加（append)している
                chart.series.append(ser)
            ws1.add_chart(chart, "H2")
            #最後にデータを全部詰め込んで完成したグラフをエクセルシート開いているスペースに貼り付けている

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
        #普通はファオルを保存する時ハードディスクに保存するがメモリの中に特殊な入れ物を用意している。ブラウザではたびたびこの形が使われる
        wb.save(output)
        output.seek(0)
        #ファイルの読み込み位置を一番最初に戻すような処理
        
        raw_filename = f"universal_mi_report_{target_project}.xlsx" if target_project != 'すべて' else "universal_mi_report.xlsx"
        #もし特定のプロジェクトが選ばれているならばその名前を入れ全体なら自動的に組み立てる
        encoded_filename = quote(raw_filename)

        # ファイル名に日本語や特殊な記号が入った時にブラウザによって文字化けしたりダウンロードエラーが起きたりするのを防ぐため安全な文字コード変換することができる
        
        return Response(
            output.getvalue(),
            #メモリ上に溜め込んだエクセルのバイナリデータを取り出して渡している
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            #これはエクセルファイルであるという正しい目印を貼っている
            headers={"Content-disposition": f"attachment; filename*=UTF-8''{encoded_filename}"}
            #ブラウザに対して画面に表示ではなくファイルとしてダウンロードさせてねという指示
        )
    except Exception as e:
        traceback.print_exc()
        return "Excel生成エラー", 500
@app.route("/api/predict_dynamic", methods=["POST"])
#ブラウザから/api/predict/dynamicというアドれエスに対してデータを送るというリクエストが飛んできた時に実行する
def predict_dynamic():
    try:
        # 1. 画面またはリクエストから予測したい「仮の条件」を受け取る
        req_data = request.get_json() or {}
        target_temp = float(req_data.get("temperature", 25.0)) # 予測したい温度（例: 25℃）
        target_mat_a = float(req_data.get("material_a", 1.5))  # 予測したい物質Aの量（例: 1.5g）
        target_mat_b = float(req_data.get("material_b", 0.5))  # 予測したい物質Bの量（例: 0.5g）

        # 2. データベース(MongoDB)から、学習元となる時系列データを引っ張る
        # ここでは、よしがアップロードした最新のCSVデータを1件取得して学習元とする
        latest_spec = db.spectrums.find_one(sort=[('_id', -1)])
        if not latest_spec:
            return {"error": "データベースに学習元のCSVデータがありません。"}, 400

        x_vals = latest_spec.get('x_values', []) # これが時間 t (Time sec)
        y_vals = latest_spec.get('y_values', []) # これが応答 (Current mA など)

        if len(x_vals) < 5:
            return {"error": "学習させるためのデータ点が足りません(最低5点以上必要です)。"}, 400

        # 3. 【ここがMIの心臓部】時系列データに「実験条件」をマージしてAI用の学習データ(X)を作る
        # 本来はいろんな実験条件が混ざるが、まずは「温度=25℃, A=1.5g, B=0.5g」でこの挙動をした、とAIに教え込む
        X_train = []
        y_train = []
        for t, val in zip(x_vals, y_vals):
            # [時間t, 温度, 物質A, 物質B] という多次元の入力ベクトルを作る
            X_train.append([t, 25.0, 1.5, 0.5]) 
            y_train.append(val)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        
        kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0, 10.0, 1.0, 1.0], (1e-2, 1e2)) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.0)
        # 4. ガウス過程回帰（ベイズ）モデルの定義
        # 時間変化と環境温度の非線形なうねりを捉えるためのカーネル設計
        

        gp.fit(X_train, y_train)
        #gpというAIの道具に対してfitという学習させるという命令をしている

        # 5. 「もし指定された条件だったらどうなる？」をシミュレーション予測する
        # 時間 t を 0 から 元のデータの最大時間まで、100分割した細かい時間ステップで予測させる
        max_time = max(x_vals)
        predict_times = np.linspace(0, max_time, 100)
        
        # 予測させたい入力データ（指定された温度、指定された物質量、各時間t）を100行分作る
        X_predict = []
        for pt in predict_times:
            X_predict.append([pt, target_temp, target_mat_a, target_mat_b])
        X_predict = np.array(X_predict)

        # 予測を実行（y_std を True にすることで、ベイズ特有の「自信のなさ（標準偏差）」も同時に計算できる）
        y_pred, y_std = gp.predict(X_predict, return_std=True)

        # 6. フロントエンド（Chart.js）でウネウネ描画できるようにJSONに整形して返す
        response_data = {
            "times": predict_times.tolist(),
            "predictions": y_pred.tolist(),
            "confidence_high": (y_pred + 1.96 * y_std).tolist(), # 95%信頼区間の上限
            "confidence_low": (y_pred - 1.96 * y_std).tolist(),  # 95%信頼区間の下限
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
    #今動かしているこのファイルがプログラム全体の主役である場合だけ下の処理をこせという書き方
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
    #普通のFlaskアプリではなくリアルタイムつしんができるsocktIOの機能も含めた状態でサーバーを起動する命令
    #debug=Trueでプログラムを書き換えて保存した際に自動でサーバーが再起動してくれるようなモード
    #パソコンの5000番ポートを使用して外部からのアクセスを受け付けるようにサーバーをスタンバイさせる設定