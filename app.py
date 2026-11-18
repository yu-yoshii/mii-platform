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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
# 機械学習の機能、ガウス過程回帰を読み出している

app = Flask(__name__)
# Flask（）でウェブアプリの仕組みを実体化している
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
MAX_GPR_TRAINING_POINTS = 1000
DEFAULT_GPR_TRAINING_POINTS = 250
MAX_VALIDATION_POINTS = 300
MIN_EXPERIMENT_POINTS = 5

MODEL_LABELS = {
    "gpr": "ガウス過程回帰（主モデル）",
    "polynomial": "2次多項式回帰",
    "random_forest": "ランダムフォレスト回帰"
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

def fit_experiment_candidate(
    model_type,
    input_values,
    target_values,
    input_columns,
    target_columns,
    optimize_gpr=True
):
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
        return {
            "type": model_type,
            "scaler": scaler,
            "polynomial": polynomial,
            "target_scaler": target_scaler,
            "estimator": estimator
        }
    if model_type == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1
        )
        forest_targets = target_values[:, 0] if target_values.shape[1] == 1 else target_values
        estimator.fit(scaled_values, forest_targets)
        return {
            "type": model_type,
            "scaler": scaler,
            "estimator": estimator
        }
    raise ValueError("指定した予測モデルは利用できません。")


def predict_experiment_candidate(model, input_values):
    model_type = model["type"]
    scaled_values = model["scaler"].transform(input_values)
    if model_type == "gpr":
        mean, std = model["estimator"].predict(scaled_values, return_std=True)
        mean = np.asarray(mean)
        std = np.asarray(std)
        if mean.ndim == 1:
            mean = mean.reshape(-1, 1)
            std = std.reshape(-1, 1)
        return mean, std
    if model_type == "random_forest":
        mean = np.asarray(model["estimator"].predict(scaled_values))
        if mean.ndim == 1:
            mean = mean.reshape(-1, 1)
        return mean, None

    features = model["polynomial"].transform(scaled_values)
    scaled_mean = model["estimator"].predict(features)
    mean = model["target_scaler"].inverse_transform(
        np.asarray(scaled_mean).reshape(len(input_values), -1)
    )
    return mean, None


def experiment_run_key(run, index):
    source_key = run.get("source_key")
    if source_key is not None:
        return str(source_key)
    if run.get("csv_file_id") is not None:
        return str(run["csv_file_id"])
    if run.get("run_id") is not None:
        return str(run["run_id"])
    return str(index)


def load_experiment_runs(spec):
    input_columns = spec.get("input_columns")
    target_columns = spec.get("target_columns")
    runs = spec.get("runs", [])
    if input_columns and target_columns and runs:
        loaded_runs = []
        for index, run in enumerate(runs):
            if "input_values" in run and "target_values" in run:
                input_values = np.asarray(run["input_values"], dtype=float)
                target_values = np.asarray(run["target_values"], dtype=float)
            else:
                csv_bytes = fs.get(run["csv_file_id"]).read()
                headers, rows = parse_csv_table(csv_bytes)
                input_values, target_values = parse_numeric_columns(
                    headers, rows, input_columns, target_columns, minimum_rows=1
                )
            loaded_runs.append({
                "input_values": input_values,
                "target_values": target_values,
                "row_count": len(input_values),
                "source_key": experiment_run_key(run, index),
                "label": run.get("label", run.get("csv_filename", f"CSV {index + 1}"))
            })
        return loaded_runs

    input_columns, target_columns, input_values, target_values = load_experiment_table(spec)
    return [{
        "input_values": input_values,
        "target_values": target_values,
        "row_count": len(input_values),
        "source_key": str(spec["_id"]),
        "label": spec.get("experiment_name", spec.get("material_name", "実験データ"))
    }]


def build_training_experiment(experiment_ids):
    specs = []
    for experiment_id in experiment_ids:
        if not isinstance(experiment_id, str) or not ObjectId.is_valid(experiment_id):
            raise ValueError("学習対象のデータを選び直してください。")
        spec = find_experiment(ObjectId(experiment_id))
        if spec is None:
            raise ValueError("選択したデータが見つかりません。")
        specs.append(spec)
    if not specs:
        raise ValueError("学習対象のデータを1つ以上選択してください。")

    first = specs[0]
    input_columns = first.get("input_columns")
    target_columns = first.get("target_columns")
    if not input_columns or not target_columns:
        input_columns, target_columns, _, _ = load_experiment_table(first)
    project_name = first.get("project_name", "未分類")
    combined_runs = []
    for spec in specs:
        columns = spec.get("input_columns"), spec.get("target_columns")
        if not all(columns):
            source_inputs, source_targets, _, _ = load_experiment_table(spec)
            columns = source_inputs, source_targets
        if columns != (input_columns, target_columns):
            raise ValueError("まとめて学習するデータは条件列と結果列の構成をそろえてください。")
        if len(specs) > 1 and spec.get("project_name", "未分類") != project_name:
            raise ValueError("まとめて学習するデータは同じプロジェクトから選択してください。")

        source_id = str(spec["_id"])
        source_name = spec.get("experiment_name", spec.get("material_name", "実験データ"))
        if spec.get("runs"):
            for index, run in enumerate(spec["runs"]):
                combined_run = dict(run)
                combined_run["source_key"] = f"{source_id}:{experiment_run_key(run, index)}"
                combined_run["label"] = (
                    f"{source_name} / {run.get('csv_filename', f'CSV {index + 1}')}"
                )
                combined_runs.append(combined_run)
        else:
            loaded_run = load_experiment_runs(spec)[0]
            combined_runs.append({
                "input_values": loaded_run["input_values"],
                "target_values": loaded_run["target_values"],
                "row_count": loaded_run["row_count"],
                "source_key": f"{source_id}:{loaded_run['source_key']}",
                "label": source_name
            })

    return {
        "_id": ":".join(str(spec["_id"]) for spec in specs),
        "source_ids": [str(spec["_id"]) for spec in specs],
        "project_name": project_name,
        "experiment_name": "、".join(
            spec.get("experiment_name", spec.get("material_name", "実験データ"))
            for spec in specs
        ),
        "input_columns": input_columns,
        "target_columns": target_columns,
        "runs": combined_runs,
        "row_count": sum(run.get("row_count", 0) for run in combined_runs),
        "run_count": len(combined_runs)
    }


def request_experiment_ids(req_data):
    experiment_ids = req_data.get("experiment_ids")
    if experiment_ids is None:
        experiment_ids = [req_data.get("experiment_id")]
    if (
        not isinstance(experiment_ids, list)
        or not experiment_ids
        or len(experiment_ids) > 50
        or any(not isinstance(value, str) or not ObjectId.is_valid(value)
               for value in experiment_ids)
    ):
        raise ValueError("学習対象のデータを1〜50件選択してください。")
    if len(set(experiment_ids)) != len(experiment_ids):
        raise ValueError("同じデータを重複して選択できません。")
    return experiment_ids


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
        experiment_run_key(run, index)
        for index, run in enumerate(spec.get("runs", []))
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

    candidate_types = ["gpr", "polynomial", "random_forest"]

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
        experiment_run_key(run, index)
        for index, run in enumerate(spec.get("runs", []))
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


def gaussian_prediction_interval(mean_values, std_values):
    if std_values is None:
        return None
    return {
        "level": 0.95,
        "method": "gaussian_process_posterior_predictive",
        "lower": np.asarray(mean_values) - 1.96 * np.asarray(std_values),
        "upper": np.asarray(mean_values) + 1.96 * np.asarray(std_values)
    }


def prediction_model_metadata(model_type, selection_reason, validation):
    metadata = {
        "model_type": model_type,
        "model_label": MODEL_LABELS[model_type],
        "model_selection_reason": selection_reason,
        "validation_available": validation["available"],
        "validation_runs": validation["fold_count"],
        "model_evaluation": validation["candidates"]
    }
    return metadata


def load_experiment_table(spec):
    input_columns = spec.get("input_columns")
    target_columns = spec.get("target_columns")
    source_files = tuple(
        experiment_run_key(run, index)
        for index, run in enumerate(spec.get("runs", []))
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
        loaded_runs = load_experiment_runs(spec)
        input_values = np.concatenate(
            [run["input_values"] for run in loaded_runs], axis=0
        )
        target_values = np.concatenate(
            [run["target_values"] for run in loaded_runs], axis=0
        )
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
                run_id = str(spec["_id"]) if spec.get("csv_file_id") else None
                csv_filename = spec.get("csv_filename", "CSV")
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
                run_id = None
                csv_filename = None
                message = f"【実験データ】{data_points} 点の {x_lbl} / {y_lbl} データを保持"

            timeline_item = {
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
            }
            if run_id:
                timeline_item.update({
                    "run_id": run_id,
                    "csv_filename": csv_filename,
                    "group_row_count": data_points
                })
            combined_timeline.append(timeline_item)

        groups = getattr(db, "experiment_groups", None)
        if groups is not None:
            for group in groups.find().sort("date", -1).limit(50):
                input_columns = group.get("input_columns", [])
                target_columns = group.get("target_columns", [])
                runs = group.get("runs", [])
                for index, run in enumerate(runs):
                    row_count = run.get("row_count", 0)
                    combined_timeline.append({
                        "id": str(group["_id"]),
                        "run_id": str(
                            run.get("run_id", run.get("csv_file_id", index))
                        ),
                        "date": run.get("date", group.get("date", datetime.now())),
                        "project_name": group.get("project_name", "未分類"),
                        "material_name": group.get("material_name", "名称不明"),
                        "domain": group.get("domain", "未分類"),
                        "experiment_name": group.get("experiment_name", "実験データ"),
                        "csv_filename": run.get("csv_filename", f"CSV {index + 1}"),
                        "message": (
                            f"【{group.get('domain', '実験データ')}】"
                            f"{run.get('csv_filename', f'CSV {index + 1}')} / "
                            f"{row_count}行（セット合計{group.get('row_count', row_count)}行）"
                        ),
                        "h_val": "N/A", "s_val": "N/A", "v_val": "N/A",
                        "image_data": "",
                        "type": "csv",
                        "row_count": row_count,
                        "group_row_count": group.get("row_count", row_count),
                        "run_count": len(runs),
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


@app.route("/api/experiments/<experiment_id>/runs/<run_id>")
def get_experiment_run(experiment_id, run_id):
    if not ObjectId.is_valid(experiment_id):
        return {"error": "実験データを選び直してください。"}, 400
    spec = find_experiment(ObjectId(experiment_id))
    if not spec:
        return {"error": "選択した実験データが見つかりません。"}, 404

    input_columns = spec.get("input_columns", [])
    target_columns = spec.get("target_columns", [])
    if not spec.get("runs") and spec.get("csv_file_id"):
        if run_id not in {str(spec["_id"]), str(spec["csv_file_id"])}:
            return {"error": "CSVデータが見つかりません。"}, 404
        try:
            csv_bytes = fs.get(spec["csv_file_id"]).read()
            headers, rows = parse_csv_table(csv_bytes)
            input_values, target_values = parse_numeric_columns(
                headers, rows, input_columns, target_columns, minimum_rows=1
            )
            return {
                "experiment_name": spec.get("experiment_name", "実験データ"),
                "filename": spec.get("csv_filename", "CSV"),
                "input_columns": input_columns,
                "target_columns": target_columns,
                "input_values": input_values.tolist(),
                "target_values": target_values.tolist(),
                "row_count": len(input_values)
            }
        except (KeyError, ValueError, UnicodeDecodeError, csv.Error):
            traceback.print_exc()
            return {"error": "CSVのグラフ用データを読み込めませんでした。"}, 500
    run = next((
        candidate for candidate in spec.get("runs", [])
        if experiment_run_key(candidate, 0) == run_id
        or str(candidate.get("run_id", "")) == run_id
    ), None)
    if run is None:
        return {"error": "CSVデータが見つかりません。"}, 404
    try:
        if "input_values" in run and "target_values" in run:
            input_values = np.asarray(run["input_values"], dtype=float)
            target_values = np.asarray(run["target_values"], dtype=float)
        else:
            csv_bytes = fs.get(run["csv_file_id"]).read()
            headers, rows = parse_csv_table(csv_bytes)
            input_values, target_values = parse_numeric_columns(
                headers, rows, input_columns, target_columns, minimum_rows=1
            )
        return {
            "experiment_name": spec.get("experiment_name", "実験データ"),
            "filename": run.get("csv_filename", "CSV"),
            "input_columns": input_columns,
            "target_columns": target_columns,
            "input_values": input_values.tolist(),
            "target_values": target_values.tolist(),
            "row_count": len(input_values)
        }
    except (KeyError, ValueError, UnicodeDecodeError, csv.Error):
        traceback.print_exc()
        return {"error": "CSVのグラフ用データを読み込めませんでした。"}, 500


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
            "run_id": str(ObjectId()),
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
            "run_id": run["run_id"],
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
            run_id = data.get("run_id")
            runs = target_group.get("runs", [])
            if run_id:
                target_run = next((
                    run for index, run in enumerate(runs)
                    if str(run.get("run_id", run.get("csv_file_id", index))) == run_id
                ), None)
                if target_run is None:
                    return
                fs.delete(target_run["csv_file_id"])
                remaining_runs = [run for run in runs if run is not target_run]
                if remaining_runs:
                    remaining_row_count = sum(
                        run.get("row_count", 0) for run in remaining_runs
                    )
                    groups.update_one(
                        {"_id": target_group["_id"]},
                        {
                            "$set": {
                                "runs": remaining_runs,
                                "row_count": remaining_row_count,
                                "run_count": len(remaining_runs)
                            }
                        }
                    )
                else:
                    remaining_row_count = 0
                    groups.delete_one({"_id": target_group["_id"]})
                emit(
                    'message deleted',
                    {
                        'id': msg_id,
                        'run_id': run_id,
                        'group_row_count': remaining_row_count
                    },
                    broadcast=True
                )
            else:
                for run in runs:
                    fs.delete(run["csv_file_id"])
                groups.delete_one({"_id": ObjectId(msg_id)})
                emit('message deleted', {'id': msg_id}, broadcast=True)
            experiment_table_cache.update({"cache_key": None, "table": None})
            model_cache.update({"cache_key": None, "scaler": None, "gp_model": None})
            model_validation_cache.update({"cache_key": None, "result": None})
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
        try:
            experiment_ids = request_experiment_ids(req_data)
            spec = build_training_experiment(experiment_ids)
        except ValueError as error:
            return {"error": str(error)}, 400

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
                req_data.get("model_type", "gpr")
            )
        except ValueError as error:
            return {"error": str(error)}, 400
        query = np.asarray([query_values], dtype=float)
        y_pred, y_std = predict_experiment_candidate(model, query)
        prediction_values = np.asarray(y_pred).reshape(-1)
        interval = gaussian_prediction_interval(
            prediction_values.reshape(1, -1),
            None if y_std is None else np.asarray(y_std).reshape(1, -1)
        )

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
            "experiment_ids": spec["source_ids"],
            "training_run_count": spec["run_count"],
            "training_points": requested_points,
            "total_points": point_total,
            "input_conditions": dict(zip(input_columns, query_values)),
            "training_ranges": training_ranges,
            "out_of_range_conditions": out_of_range_conditions,
            **prediction_model_metadata(
                model_type, selection_reason, validation
            ),
            "predictions": dict(zip(target_columns, prediction_values.tolist())),
            "prediction_interval": None if interval is None else {
                "level": interval["level"],
                "method": interval["method"]
            },
            "confidence_high": None if interval is None else dict(zip(
                target_columns,
                np.asarray(interval["upper"]).reshape(-1).tolist()
            )),
            "confidence_low": None if interval is None else dict(zip(
                target_columns,
                np.asarray(interval["lower"]).reshape(-1).tolist()
            ))
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}, 500

@app.route("/api/predict_grid", methods=["POST"])
def predict_grid():
    try:
        req_data = request.get_json() or {}
        try:
            experiment_ids = request_experiment_ids(req_data)
            spec = build_training_experiment(experiment_ids)
        except ValueError as error:
            return {"error": str(error)}, 400

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
                req_data.get("model_type", "gpr")
            )
        except ValueError as error:
            return {"error": str(error)}, 400
        predicted, predicted_std = predict_experiment_candidate(model, queries)
        prediction_values = np.asarray(predicted)
        std_values = None if predicted_std is None else np.asarray(predicted_std)

        if prediction_values.ndim == 1:
            prediction_values = prediction_values.reshape(-1, 1)
            if std_values is not None:
                std_values = std_values.reshape(-1, 1)
        if y_values is not None:
            prediction_values = prediction_values.reshape(
                grid_size, grid_size, len(target_columns)
            )
            if std_values is not None:
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
                model_type, selection_reason, validation
            ),
            "experiment_ids": spec["source_ids"],
            "training_run_count": spec["run_count"],
            "fixed_conditions": {
                column: float(fixed_values[index])
                for index, column in enumerate(input_columns)
                if column not in (x_column, y_column)
            },
            "prediction_interval": None if std_values is None else {
                "level": 0.95,
                "method": "gaussian_process_posterior_predictive"
            }
        }
        for target_index, target_column in enumerate(target_columns):
            mean_values = prediction_values[..., target_index]
            response.setdefault("predictions", {})[target_column] = mean_values.tolist()
            if std_values is not None:
                interval = gaussian_prediction_interval(
                    mean_values, std_values[..., target_index]
                )
                response.setdefault("confidence_low", {})[target_column] = (
                    interval["lower"].tolist()
                )
                response.setdefault("confidence_high", {})[target_column] = (
                    interval["upper"].tolist()
                )

        measured_series = []
        source_runs = load_experiment_runs(spec)
        max_points_per_run = max(1, 2000 // len(source_runs))
        y_index = input_columns.index(y_column) if y_column is not None else None
        target_index = target_columns.index(req_data.get("target_column")) \
            if req_data.get("target_column") in target_columns else 0
        for run in source_runs:
            run_inputs = run["input_values"]
            run_targets = run["target_values"]
            if len(run_inputs) > max_points_per_run:
                run_inputs, run_targets = select_evenly_spaced_rows(
                    run_inputs, run_targets, max_points_per_run
                )
            series = {
                "label": run["label"],
                "x_values": run_inputs[:, x_index].tolist(),
                "target_values": run_targets[:, target_index].tolist()
            }
            if y_index is not None:
                series["y_values"] = run_inputs[:, y_index].tolist()
            measured_series.append(series)
        response["measured_data"] = measured_series
        if req_data.get("compare_models") and y_column is None:
            comparison_predictions = {}
            x_train, y_train = select_evenly_spaced_rows(
                input_values, target_values, requested_points
            )
            for candidate_type in MODEL_LABELS:
                candidate_model = model if candidate_type == model_type else fit_experiment_candidate(
                    candidate_type,
                    x_train,
                    y_train,
                    input_columns,
                    target_columns,
                    optimize_gpr=False
                )
                candidate_values, _ = predict_experiment_candidate(
                    candidate_model, queries
                )
                candidate_values = np.asarray(candidate_values).reshape(
                    grid_size, len(target_columns)
                )
                comparison_predictions[candidate_type] = {
                    column: candidate_values[:, index].tolist()
                    for index, column in enumerate(target_columns)
                }
            response["model_comparison_predictions"] = comparison_predictions
            response["model_labels"] = MODEL_LABELS
        return response
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}, 500

if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)