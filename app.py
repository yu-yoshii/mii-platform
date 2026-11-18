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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from scipy.optimize import least_squares
# 機械学習の機能、ガウス過程回帰を読み出している

app = Flask(__name__)
# Flask（）でウェブアプリの仕組みを実体化している
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
MAX_GPR_TRAINING_POINTS = 1000
DEFAULT_GPR_TRAINING_POINTS = 250
MAX_VALIDATION_POINTS = 300
MIN_EXPERIMENT_POINTS = 5

MODEL_LABELS = {
    "gpr": "汎用ガウス過程（補間向け）",
    "polynomial": "2次多項式回帰",
    "rc_parallel": "並列RC回路モデル"
}

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

def parse_numeric_columns(headers, rows, input_columns, target_columns, minimum_rows=5):
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

    if len(numeric_rows) < minimum_rows:
        raise ValueError(f"測定データには{minimum_rows}行以上必要です。")

    values = np.asarray(numeric_rows, dtype=float)
    return values[:, :len(input_columns)], values[:, len(input_columns):]

def select_evenly_spaced_rows(input_values, target_values, row_count):
    indices = np.linspace(0, len(input_values) - 1, num=row_count, dtype=int)
    return np.asarray(input_values)[indices], np.asarray(target_values)[indices]

def resolve_axis_range(requested_range, axis_values):
    observed_min = float(np.min(axis_values))
    observed_max = float(np.max(axis_values))
    if requested_range is None:
        if observed_min == observed_max:
            padding = max(abs(observed_min) * 0.1, 0.5)
            return observed_min - padding, observed_max + padding
        return observed_min, observed_max
    if not isinstance(requested_range, list) or len(requested_range) != 2:
        raise ValueError("グラフの範囲は最小値と最大値を指定してください。")
    if any(isinstance(value, bool) for value in requested_range):
        raise ValueError("グラフの範囲は数値で指定してください。")
    try:
        low, high = map(float, requested_range)
    except (TypeError, ValueError) as error:
        raise ValueError("グラフの範囲は数値で指定してください。") from error
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError("グラフ範囲の最小値は最大値より小さくしてください。")
    return low, high

def find_rc_columns(input_columns, target_columns):
    def normalized(column):
        return "".join(character for character in column.lower() if character.isalnum())

    current_columns = [
        index for index, column in enumerate(input_columns)
        if "current" in normalized(column)
        or "ampere" in normalized(column)
        or "電流" in normalized(column)
        or normalized(column).startswith("amps")
        or normalized(column) in {"amp", "ia", "iamp"}
    ]
    time_columns = [
        index for index, column in enumerate(input_columns)
        if any(token in normalized(column) for token in ("time", "elapsed", "時間"))
        or normalized(column) in {"t", "ts", "tsec", "秒"}
    ]
    voltage_columns = [
        index for index, column in enumerate(target_columns)
        if any(token in normalized(column) for token in ("voltage", "volt", "電圧"))
        or normalized(column) in {"v", "vc", "vout"}
    ]
    if (
        len(current_columns) != 1
        or len(time_columns) != 1
        or len(voltage_columns) != 1
        or len(target_columns) != 1
    ):
        return None
    return current_columns[0], time_columns[0], voltage_columns[0]


def fit_rc_parallel_model(input_values, target_values, input_columns, target_columns):
    indices = find_rc_columns(input_columns, target_columns)
    if indices is None:
        raise ValueError(
            "RC回路モデルには、条件列の電流・時間と結果列の電圧を識別できる列名が必要です。"
        )
    if target_values.shape[1] != 1 or len(input_values) < 5:
        raise ValueError("RC回路モデルは電圧1列、測定データ5行以上で使用できます。")

    current_index, time_index, voltage_index = indices
    current = input_values[:, current_index]
    time = input_values[:, time_index]
    voltage = target_values[:, voltage_index]
    if np.any(time < 0) or len(np.unique(time)) < 3 or np.allclose(current, 0):
        raise ValueError("RC回路モデルには、0以上の時間と変化のある電流データが必要です。")

    initial_offset = float(voltage[np.argmin(time)])
    current_scale = max(float(np.max(np.abs(current))), 1e-9)
    resistance_guess = max(
        float(np.ptp(voltage)) / current_scale,
        1e-6
    )
    time_constant_guess = max(float(np.ptp(time)) / 3, 1e-3)

    def calculate(parameters, current_values, time_values):
        resistance, time_constant = np.exp(parameters[:2])
        offset = parameters[2]
        return offset + current_values * resistance * (
            -np.expm1(-time_values / time_constant)
        )

    fit = least_squares(
        lambda parameters: calculate(parameters, current, time) - voltage,
        x0=[
            np.log(resistance_guess),
            np.log(time_constant_guess),
            initial_offset
        ],
        bounds=([-30, -30, -np.inf], [30, 30, np.inf]),
        max_nfev=5000
    )
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise ValueError("RC回路の抵抗と時定数をデータから推定できませんでした。")

    residuals = calculate(fit.x, current, time) - voltage
    degrees_of_freedom = max(len(voltage) - len(fit.x), 1)
    residual_variance = float(np.dot(residuals, residuals) / degrees_of_freedom)
    covariance = residual_variance * np.linalg.pinv(fit.jac.T @ fit.jac)
    resistance, time_constant = np.exp(fit.x[:2])
    return {
        "type": "rc_parallel",
        "current_index": current_index,
        "time_index": time_index,
        "voltage_index": voltage_index,
        "resistance": float(resistance),
        "time_constant": float(time_constant),
        "offset": float(fit.x[2]),
        "covariance": covariance,
        "residual_variance": residual_variance
    }


def fit_experiment_candidate(
    model_type,
    input_values,
    target_values,
    input_columns,
    target_columns,
    optimize_gpr=True
):
    if model_type == "rc_parallel":
        return fit_rc_parallel_model(
            input_values, target_values, input_columns, target_columns
        )

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(input_values)
    if model_type == "gpr":
        kernel = (
            C(1.0, (1e-3, 1e3))
            * RBF(np.ones(len(input_columns)), (1e-2, 1e2))
            + WhiteKernel(noise_level=0.1)
        )
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=1 if optimize_gpr else 0,
            alpha=1e-6,
            normalize_y=True
        )
        model.fit(scaled_values, target_values)
        return {
            "type": model_type,
            "scaler": scaler,
            "estimator": model
        }
    if model_type == "polynomial":
        polynomial = PolynomialFeatures(degree=2, include_bias=False)
        features = polynomial.fit_transform(scaled_values)
        target_scaler = StandardScaler()
        scaled_targets = target_scaler.fit_transform(target_values)
        estimator = Ridge(alpha=1.0)
        estimator.fit(features, scaled_targets)
        residuals = target_values - target_scaler.inverse_transform(
            np.asarray(estimator.predict(features)).reshape(len(features), -1)
        )
        residual_std = np.sqrt(np.mean(np.square(residuals), axis=0))
        return {
            "type": model_type,
            "scaler": scaler,
            "polynomial": polynomial,
            "target_scaler": target_scaler,
            "estimator": estimator,
            "residual_std": residual_std
        }
    raise ValueError("指定した予測モデルは利用できません。")


def predict_experiment_candidate(model, input_values):
    model_type = model["type"]
    if model_type == "rc_parallel":
        current = input_values[:, model["current_index"]]
        time = input_values[:, model["time_index"]]
        resistance = model["resistance"]
        time_constant = model["time_constant"]
        decay = np.exp(-time / time_constant)
        amplitude = current * resistance
        response = amplitude * (1 - decay)
        mean = (model["offset"] + response).reshape(-1, 1)
        gradient = np.column_stack((
            response,
            -amplitude * decay * time / time_constant,
            np.ones(len(time))
        ))
        parameter_variance = np.einsum(
            "ij,jk,ik->i",
            gradient,
            model["covariance"],
            gradient
        )
        std = np.sqrt(np.maximum(
            parameter_variance + model["residual_variance"],
            0
        )).reshape(-1, 1)
        return mean, std

    scaled_values = model["scaler"].transform(input_values)
    if model_type == "gpr":
        mean, std = model["estimator"].predict(scaled_values, return_std=True)
        mean = np.asarray(mean)
        std = np.asarray(std)
        if mean.ndim == 1:
            mean = mean.reshape(-1, 1)
            std = std.reshape(-1, 1)
        return mean, std

    features = model["polynomial"].transform(scaled_values)
    scaled_mean = model["estimator"].predict(features)
    mean = model["target_scaler"].inverse_transform(
        np.asarray(scaled_mean).reshape(len(input_values), -1)
    )
    std = np.tile(model["residual_std"], (len(input_values), 1))
    return mean, std


def get_run_slices(spec, row_count):
    runs = spec.get("runs", [])
    run_lengths = [run.get("row_count") for run in runs]
    if len(runs) < 2 or any(
        isinstance(length, bool) or not isinstance(length, int) or length < 1
        for length in run_lengths
    ) or sum(run_lengths) != row_count:
        return []

    slices = []
    start = 0
    for length in run_lengths:
        slices.append(slice(start, start + length))
        start += length
    return slices


model_validation_cache = {"cache_key": None, "result": None}


def validate_experiment_models(
    spec,
    input_columns,
    target_columns,
    input_values,
    target_values,
    requested_points
):
    run_slices = get_run_slices(spec, len(input_values))
    if not run_slices:
        return {
            "available": False,
            "fold_count": 0,
            "candidates": [],
            "best_model": None,
            "residual_q95": {}
        }

    source_files = tuple(
        str(run["csv_file_id"]) for run in spec.get("runs", [])
    )
    cache_key = (
        str(spec["_id"]),
        source_files,
        tuple(input_columns),
        tuple(target_columns),
        requested_points
    )
    if model_validation_cache["cache_key"] == cache_key:
        return model_validation_cache["result"]

    candidate_types = ["gpr", "polynomial"]
    if find_rc_columns(input_columns, target_columns) is not None:
        candidate_types.append("rc_parallel")

    candidates = []
    target_scales = np.ptp(target_values, axis=0)
    fallback_scales = np.std(target_values, axis=0)
    target_scales = np.where(
        target_scales > 1e-12,
        target_scales,
        np.where(fallback_scales > 1e-12, fallback_scales, 1.0)
    )

    for model_type in candidate_types:
        errors = []
        fold_rmses = []
        fold_count = 0
        failed = False
        for held_out in run_slices:
            train_mask = np.ones(len(input_values), dtype=bool)
            train_mask[held_out] = False
            train_values = input_values[train_mask]
            train_targets = target_values[train_mask]
            test_values = input_values[held_out]
            test_targets = target_values[held_out]
            if len(test_values) > MAX_VALIDATION_POINTS:
                test_values, test_targets = select_evenly_spaced_rows(
                    test_values,
                    test_targets,
                    MAX_VALIDATION_POINTS
                )
            if len(train_values) < MIN_EXPERIMENT_POINTS:
                failed = True
                break

            fold_points = min(
                len(train_values),
                requested_points,
                MAX_VALIDATION_POINTS
            )
            train_values, train_targets = select_evenly_spaced_rows(
                train_values, train_targets, fold_points
            )
            try:
                model = fit_experiment_candidate(
                    model_type,
                    train_values,
                    train_targets,
                    input_columns,
                    target_columns,
                    optimize_gpr=False
                )
                predicted, _ = predict_experiment_candidate(model, test_values)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                failed = True
                break

            fold_errors = predicted - test_targets
            errors.append(fold_errors)
            fold_rmses.append(np.sqrt(np.mean(np.square(fold_errors), axis=0)))
            fold_count += 1

        if failed or fold_count != len(run_slices):
            continue
        absolute_errors = np.abs(np.concatenate(errors, axis=0))
        rmse = np.mean(fold_rmses, axis=0)
        candidates.append({
            "model_type": model_type,
            "label": MODEL_LABELS[model_type],
            "normalized_rmse": float(np.mean(rmse / target_scales)),
            "rmse_by_target": {
                column: float(value)
                for column, value in zip(target_columns, rmse)
            },
            "residual_q95": {
                column: float(np.quantile(
                    absolute_errors[:, index],
                    0.95,
                    method="higher"
                ))
                for index, column in enumerate(target_columns)
            },
            "fold_count": fold_count
        })

    result = {
        "available": bool(candidates),
        "fold_count": len(run_slices),
        "candidates": candidates,
        "best_model": min(
            candidates,
            key=lambda candidate: candidate["normalized_rmse"]
        )["model_type"] if candidates else None,
        "residual_q95": {
            candidate["model_type"]: candidate["residual_q95"]
            for candidate in candidates
        }
    }
    model_validation_cache.update({"cache_key": cache_key, "result": result})
    return result


def get_experiment_model(
    spec,
    input_columns,
    target_columns,
    input_values,
    target_values,
    requested_points,
    model_type="gpr"
):
    source_files = tuple(
        str(run["csv_file_id"]) for run in spec.get("runs", [])
    ) or (str(spec.get("csv_file_id", "legacy")),)
    cache_key = (
        str(spec["_id"]),
        source_files,
        tuple(input_columns),
        tuple(target_columns),
        requested_points,
        model_type
    )
    if model_cache["cache_key"] != cache_key or model_cache.get("gp_model") is None:
        print(f"🔄 実験データから{MODEL_LABELS[model_type]}を学習します...")
        x_train, y_train = select_evenly_spaced_rows(
            input_values, target_values, requested_points
        )
        model = fit_experiment_candidate(
            model_type,
            x_train,
            y_train,
            input_columns,
            target_columns
        )
        model_cache["cache_key"] = cache_key
        model_cache["scaler"] = model.get("scaler")
        model_cache["gp_model"] = model
    else:
        print(f"⚡ キャッシュされた{MODEL_LABELS[model_type]}モデルを使用します")

    return model_cache["scaler"], model_cache["gp_model"]


def prepare_experiment_model(
    spec,
    input_columns,
    target_columns,
    input_values,
    target_values,
    requested_points,
    requested_model
):
    if requested_model not in {"auto", *MODEL_LABELS}:
        raise ValueError("予測モデルを選び直してください。")
    validation = validate_experiment_models(
        spec,
        input_columns,
        target_columns,
        input_values,
        target_values,
        requested_points
    )
    if requested_model == "auto":
        model_type = validation["best_model"] or "gpr"
        selection_reason = (
            "run_holdout" if validation["best_model"] else "validation_unavailable"
        )
    else:
        model_type = requested_model
        selection_reason = "user_selected"
    if model_type == "rc_parallel" and find_rc_columns(
        input_columns, target_columns
    ) is None:
        raise ValueError(
            "RC回路モデルには、条件列の電流・時間と結果列の電圧を識別できる列名が必要です。"
        )

    _, model = get_experiment_model(
        spec,
        input_columns,
        target_columns,
        input_values,
        target_values,
        requested_points,
        model_type
    )
    return model_type, selection_reason, model, validation


def include_validation_uncertainty(std_values, validation, model_type, target_columns):
    residual_q95 = validation.get("residual_q95", {}).get(model_type)
    if not residual_q95:
        return std_values
    calibration = np.asarray(
        [residual_q95[column] for column in target_columns],
        dtype=float
    ) / 1.96
    calibration = calibration.reshape((1,) * (std_values.ndim - 1) + (-1,))
    return np.maximum(std_values, calibration)


def prediction_model_metadata(model_type, selection_reason, validation, model):
    metadata = {
        "model_type": model_type,
        "model_label": MODEL_LABELS[model_type],
        "model_selection_reason": selection_reason,
        "validation_available": validation["available"],
        "validation_runs": validation["fold_count"],
        "model_evaluation": validation["candidates"]
    }
    if model_type == "rc_parallel":
        metadata["model_parameters"] = {
            "resistance_ohm": model["resistance"],
            "time_constant_s": model["time_constant"],
            "capacitance_f": model["time_constant"] / model["resistance"]
        }
    return metadata


def load_experiment_table(spec):
    input_columns = spec.get("input_columns")
    target_columns = spec.get("target_columns")
    source_files = tuple(
        str(run["csv_file_id"]) for run in spec.get("runs", [])
    ) or (str(spec.get("csv_file_id", "legacy")),)
    cache_key = (
        str(spec["_id"]),
        source_files,
        tuple(input_columns or []),
        tuple(target_columns or []),
        spec.get("x_label"),
        spec.get("y_label")
    )
    if experiment_table_cache["cache_key"] == cache_key:
        return experiment_table_cache["table"]

    if input_columns and target_columns and spec.get("runs"):
        input_tables = []
        target_tables = []
        for run in spec["runs"]:
            csv_bytes = fs.get(run["csv_file_id"]).read()
            headers, rows = parse_csv_table(csv_bytes)
            input_values, target_values = parse_numeric_columns(
                headers, rows, input_columns, target_columns, minimum_rows=1
            )
            input_tables.append(input_values)
            target_tables.append(target_values)
        input_values = np.concatenate(input_tables, axis=0)
        target_values = np.concatenate(target_tables, axis=0)
        table = (input_columns, target_columns, input_values, target_values)
    elif input_columns and target_columns and spec.get("csv_file_id"):
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

def find_experiment(experiment_id):
    group_collection = getattr(db, "experiment_groups", None)
    if group_collection is not None:
        group = group_collection.find_one({"_id": experiment_id})
        if group:
            return group
    return db.spectrums.find_one({"_id": experiment_id})

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

        groups = getattr(db, "experiment_groups", None)
        if groups is not None:
            for group in groups.find().sort("date", -1).limit(50):
                input_columns = group.get("input_columns", [])
                target_columns = group.get("target_columns", [])
                row_count = group.get("row_count", 0)
                run_count = group.get("run_count", len(group.get("runs", [])))
                combined_timeline.append({
                    "id": str(group["_id"]),
                    "date": group.get("date", datetime.now()),
                    "project_name": group.get("project_name", "未分類"),
                    "material_name": group.get("material_name", "名称不明"),
                    "domain": group.get("domain", "未分類"),
                    "experiment_name": group.get("experiment_name", "実験データ"),
                    "message": (
                        f"【{group.get('domain', '実験データ')}】{run_count}件のCSV"
                        f" / {row_count}行 / 条件: {', '.join(input_columns)}"
                        f" / 結果: {', '.join(target_columns)}"
                    ),
                    "h_val": "N/A", "s_val": "N/A", "v_val": "N/A",
                    "image_data": "",
                    "type": "csv",
                    "row_count": row_count,
                    "run_count": run_count,
                    "input_columns": input_columns,
                    "target_columns": target_columns,
                    "x_label": ", ".join(input_columns),
                    "y_label": ", ".join(target_columns),
                    "x_values": [],
                    "y_values": [],
                    "wavelengths": [],
                    "intensities": []
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
        groups = getattr(db, "experiment_groups", None)
        if groups is not None:
            for group in groups.find().sort("date", -1):
                date = group.get("date")
                experiments.append({
                    "id": str(group["_id"]),
                    "experiment_name": group.get("experiment_name", "実験データ"),
                    "project_name": group.get("project_name", "未分類"),
                    "material_name": group.get("material_name", "名称不明"),
                    "domain": group.get("domain", "未分類"),
                    "date": date.isoformat() if isinstance(date, datetime) else str(date or ""),
                    "input_columns": group.get("input_columns", []),
                    "target_columns": group.get("target_columns", []),
                    "point_count": group.get("row_count", 0),
                    "run_count": group.get("run_count", len(group.get("runs", [])))
                })

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
                "experiment_name": spec.get("material_name", "旧形式の実験"),
                "project_name": spec.get("project_name", "未分類"),
                "material_name": spec.get("material_name", "名称不明"),
                "domain": spec.get("domain", "未分類"),
                "date": date.isoformat() if isinstance(date, datetime) else str(date or ""),
                "input_columns": input_columns,
                "target_columns": target_columns,
                "point_count": point_count,
                "run_count": 1
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
        if not isinstance(input_columns, list) or not isinstance(target_columns, list):
            return {"error": "条件列と結果列を選択してください。"}, 400
        if any(not isinstance(column, str) for column in input_columns + target_columns):
            return {"error": "列名の指定が正しくありません。"}, 400
        project_name = request.form.get("project_name", "未分類").strip() or "未分類"
        material_name = request.form.get("material_name", "名称不明").strip() or "名称不明"
        domain = request.form.get("domain", "未分類").strip() or "未分類"
        experiment_name = (
            request.form.get("experiment_name", "").strip()
            or Path(uploaded_file.filename).stem
            or "実験データ"
        )
        groups = db.experiment_groups
        group_filter = {
            "project_name": project_name,
            "material_name": material_name,
            "domain": domain,
            "experiment_name": experiment_name
        }
        group = groups.find_one(group_filter)
        if group:
            if (
                set(input_columns) != set(group.get("input_columns", []))
                or set(target_columns) != set(group.get("target_columns", []))
            ):
                return {
                    "error": (
                        "同じ実験セットに追加するCSVは、条件列と結果列を"
                        "前回と同じにしてください。"
                    )
                }, 400
            input_columns = group["input_columns"]
            target_columns = group["target_columns"]

        input_values, _ = parse_numeric_columns(
            headers, rows, input_columns, target_columns, minimum_rows=1
        )
        experiment_id = group["_id"] if group else ObjectId()
        file_id = fs.put(
            csv_bytes,
            filename=uploaded_file.filename,
            content_type="text/csv",
            metadata={"experiment_id": str(experiment_id), "kind": "experiment_data"}
        )
        run = {
            "csv_file_id": file_id,
            "csv_filename": uploaded_file.filename,
            "row_count": len(input_values),
            "date": datetime.now()
        }
        try:
            if group:
                total_rows = group.get("row_count", 0) + len(input_values)
                run_count = group.get("run_count", 1) + 1
                updated = groups.update_one(
                    {"_id": experiment_id},
                    {
                        "$push": {"runs": run},
                        "$inc": {"row_count": len(input_values), "run_count": 1},
                        "$set": {"date": run["date"]}
                    }
                )
                if updated.matched_count != 1:
                    raise RuntimeError("実験セットが更新中に見つからなくなりました。")
            else:
                groups.insert_one({
                    "_id": experiment_id,
                    **group_filter,
                    "input_columns": input_columns,
                    "target_columns": target_columns,
                    "runs": [run],
                    "row_count": len(input_values),
                    "run_count": 1,
                    "date": run["date"]
                })
                total_rows = len(input_values)
                run_count = 1
        except Exception:
            fs.delete(file_id)
            raise

        return {
            "id": str(experiment_id),
            "experiment_name": experiment_name,
            "input_columns": input_columns,
            "target_columns": target_columns,
            "row_count": total_rows,
            "added_row_count": len(input_values),
            "run_count": run_count
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

        groups = getattr(db, "experiment_groups", None)
        target_group = (
            groups.find_one({"_id": ObjectId(msg_id)})
            if groups is not None else None
        )
        if target_group:
            for run in target_group.get("runs", []):
                fs.delete(run["csv_file_id"])
            groups.delete_one({"_id": ObjectId(msg_id)})
            if experiment_table_cache["cache_key"] and experiment_table_cache["cache_key"][0] == msg_id:
                experiment_table_cache.update({"cache_key": None, "table": None})
            if model_cache["cache_key"] and model_cache["cache_key"][0] == msg_id:
                model_cache.update({"cache_key": None, "scaler": None, "gp_model": None})
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

        spec = find_experiment(ObjectId(experiment_id))
        if not spec:
            return {"error": "選択した実験データが見つかりません。"}, 404

        input_columns, target_columns, input_values, target_values = load_experiment_table(spec)
        point_total = len(input_values)
        if point_total < MIN_EXPERIMENT_POINTS:
            return {"error": "機械学習には合計5行以上の測定データが必要です。"}, 400

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

        try:
            model_type, selection_reason, model, validation = prepare_experiment_model(
                spec,
                input_columns,
                target_columns,
                input_values,
                target_values,
                requested_points,
                req_data.get("model_type", "auto")
            )
        except ValueError as error:
            return {"error": str(error)}, 400
        query = np.asarray([query_values], dtype=float)
        y_pred, y_std = predict_experiment_candidate(model, query)
        prediction_values = np.asarray(y_pred).reshape(-1)
        std_values = include_validation_uncertainty(
            np.asarray(y_std),
            validation,
            model_type,
            target_columns
        ).reshape(-1)

        training_ranges = {
            column: [
                float(np.min(input_values[:, index])),
                float(np.max(input_values[:, index]))
            ]
            for index, column in enumerate(input_columns)
        }
        out_of_range_conditions = {
            column: value
            for column, value in zip(input_columns, query_values)
            if value < training_ranges[column][0] or value > training_ranges[column][1]
        }
        return {
            "experiment_id": str(spec["_id"]),
            "training_points": requested_points,
            "total_points": point_total,
            "input_conditions": dict(zip(input_columns, query_values)),
            "training_ranges": training_ranges,
            "out_of_range_conditions": out_of_range_conditions,
            **prediction_model_metadata(
                model_type, selection_reason, validation, model
            ),
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

        spec = find_experiment(ObjectId(experiment_id))
        if not spec:
            return {"error": "選択した実験データが見つかりません。"}, 404

        input_columns, target_columns, input_values, target_values = load_experiment_table(spec)
        point_total = len(input_values)
        if point_total < MIN_EXPERIMENT_POINTS:
            return {"error": "機械学習には合計5行以上の測定データが必要です。"}, 400
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
        try:
            x_range = resolve_axis_range(req_data.get("x_range"), input_values[:, x_index])
        except ValueError as error:
            return {"error": str(error)}, 400
        x_values = np.linspace(
            x_range[0],
            x_range[1],
            grid_size
        )
        if y_column is None:
            queries = np.tile(fixed_values, (grid_size, 1))
            queries[:, x_index] = x_values
            y_values = None
            y_range = None
        else:
            y_index = input_columns.index(y_column)
            try:
                y_range = resolve_axis_range(req_data.get("y_range"), input_values[:, y_index])
            except ValueError as error:
                return {"error": str(error)}, 400
            y_values = np.linspace(
                y_range[0],
                y_range[1],
                grid_size
            )
            x_mesh, y_mesh = np.meshgrid(x_values, y_values)
            queries = np.tile(fixed_values, (grid_size * grid_size, 1))
            queries[:, x_index] = x_mesh.ravel()
            queries[:, y_index] = y_mesh.ravel()

        try:
            model_type, selection_reason, model, validation = prepare_experiment_model(
                spec,
                input_columns,
                target_columns,
                input_values,
                target_values,
                requested_points,
                req_data.get("model_type", "auto")
            )
        except ValueError as error:
            return {"error": str(error)}, 400
        predicted, predicted_std = predict_experiment_candidate(model, queries)
        prediction_values = np.asarray(predicted)
        std_values = include_validation_uncertainty(
            np.asarray(predicted_std),
            validation,
            model_type,
            target_columns
        )

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

        training_ranges = {
            column: [
                float(np.min(input_values[:, index])),
                float(np.max(input_values[:, index]))
            ]
            for index, column in enumerate(input_columns)
        }
        out_of_range_conditions = {
            column: float(fixed_values[index])
            for index, column in enumerate(input_columns)
            if column not in (x_column, y_column)
            and (
                fixed_values[index] < training_ranges[column][0]
                or fixed_values[index] > training_ranges[column][1]
            )
        }
        out_of_range_axes = {}
        if x_range[0] < training_ranges[x_column][0] or x_range[1] > training_ranges[x_column][1]:
            out_of_range_axes[x_column] = list(x_range)
        if y_column and (
            y_range[0] < training_ranges[y_column][0]
            or y_range[1] > training_ranges[y_column][1]
        ):
            out_of_range_axes[y_column] = list(y_range)
        response = {
            "x_column": x_column,
            "y_column": y_column,
            "x_range": list(x_range),
            "y_range": list(y_range) if y_range is not None else None,
            "x_values": x_values.tolist(),
            "y_values": y_values.tolist() if y_values is not None else None,
            "training_points": requested_points,
            "total_points": point_total,
            "training_ranges": training_ranges,
            "out_of_range_conditions": out_of_range_conditions,
            "out_of_range_axes": out_of_range_axes,
            **prediction_model_metadata(
                model_type, selection_reason, validation, model
            ),
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