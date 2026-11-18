import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import app as app_module
import numpy as np
from bson.objectid import ObjectId


class FakeCursor:
    def __init__(self, documents):
        self.documents = documents

    def sort(self, field, direction):
        return FakeCursor(sorted(
            self.documents,
            key=lambda document: document.get(field),
            reverse=direction < 0
        ))

    def __iter__(self):
        return iter(self.documents)

    def limit(self, count):
        return FakeCursor(self.documents[:count])


class FakeSpectrumCollection:
    def __init__(self, documents):
        self.documents = documents

    def find(self):
        return FakeCursor(self.documents)

    def find_one(self, query):
        return next(
            (document for document in self.documents if document["_id"] == query["_id"]),
            None
        )

    def insert_one(self, document):
        self.documents.append(document)


class FakeExperimentGroupCollection:
    def __init__(self, documents=None):
        self.documents = documents or []

    def find(self, query=None):
        query = query or {}
        return FakeCursor([
            document for document in self.documents
            if all(document.get(key) == value for key, value in query.items())
        ])

    def find_one(self, query):
        return next((
            document for document in self.documents
            if all(document.get(key) == value for key, value in query.items())
        ), None)

    def insert_one(self, document):
        self.documents.append(document)

    def update_one(self, query, update):
        document = self.find_one(query)
        if not document:
            return SimpleNamespace(matched_count=0)
        for key, value in update.get("$push", {}).items():
            document.setdefault(key, []).append(value)
        for key, value in update.get("$inc", {}).items():
            document[key] = document.get(key, 0) + value
        document.update(update.get("$set", {}))
        return SimpleNamespace(matched_count=1)

    def delete_one(self, query):
        document = self.find_one(query)
        if document:
            self.documents.remove(document)


class FakeGridFS:
    def __init__(self):
        self.files = {}

    def put(self, data, **metadata):
        file_id = ObjectId()
        self.files[file_id] = data
        return file_id

    def get(self, file_id):
        return SimpleNamespace(read=lambda: self.files[file_id])

    def delete(self, file_id):
        del self.files[file_id]


class ExperimentCsvTests(unittest.TestCase):
    def setUp(self):
        self.original_instance_path = app_module.app.instance_path
        self.original_db = app_module.db
        self.original_fs = app_module.fs
        self.original_model_cache = app_module.model_cache.copy()
        self.original_model_validation_cache = app_module.model_validation_cache.copy()
        self.original_experiment_table_cache = app_module.experiment_table_cache.copy()
        self.temp_directory = tempfile.TemporaryDirectory()
        app_module.app.instance_path = self.temp_directory.name
        app_module.fs = FakeGridFS()
        app_module.experiment_table_cache = {"cache_key": None, "table": None}

    def tearDown(self):
        app_module.app.instance_path = self.original_instance_path
        app_module.db = self.original_db
        app_module.fs = self.original_fs
        app_module.model_cache = self.original_model_cache
        app_module.model_validation_cache = self.original_model_validation_cache
        app_module.experiment_table_cache = self.original_experiment_table_cache
        self.temp_directory.cleanup()

    def test_saves_and_reads_experiment_csv(self):
        path = app_module.save_experiment_csv(
            "experiment-1",
            [0, 1, 2],
            [10, 20, 30]
        )

        self.assertEqual(
            app_module.read_experiment_csv(path),
            ([0.0, 1.0, 2.0], [10.0, 20.0, 30.0])
        )
        self.assertEqual(path.parent, Path(self.temp_directory.name) / "experiments")

    def test_even_sampling_spans_full_experiment(self):
        x_values, y_values = app_module.select_evenly_spaced_points(
            list(range(9)),
            [value * 10 for value in range(9)],
            5
        )

        self.assertEqual(x_values.tolist(), [0, 2, 4, 6, 8])
        self.assertEqual(y_values.tolist(), [0, 20, 40, 60, 80])

    def test_constant_graph_axis_gets_a_usable_default_range(self):
        self.assertEqual(
            app_module.resolve_axis_range(None, np.ones(5)),
            (0.5, 1.5)
        )

    def test_gaussian_prediction_interval_uses_gpr_posterior_std_directly(self):
        interval = app_module.gaussian_prediction_interval(
            np.array([[10.0]]),
            np.array([[2.0]])
        )

        self.assertAlmostEqual(interval["lower"][0, 0], 6.08)
        self.assertAlmostEqual(interval["upper"][0, 0], 13.92)
        self.assertEqual(interval["level"], 0.95)

    def test_prediction_page_defaults_to_gpr_and_offers_model_comparison(self):
        response = app_module.app.test_client().get("/predict")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn('option value="gpr" selected', html)
        self.assertIn('value="random_forest"', html)
        self.assertIn("互換データを一括選択", html)

    def test_creates_file_for_legacy_database_experiment(self):
        spec = {
            "_id": "legacy-experiment",
            "x_values": [1, 2, 3],
            "y_values": [4, 5, 6]
        }

        path = app_module.ensure_experiment_csv(spec)

        self.assertTrue(path.is_file())
        self.assertEqual(
            app_module.read_experiment_csv(path),
            ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        )

    def test_experiment_api_lists_saved_files(self):
        spec = {
            "_id": ObjectId(),
            "x_values": [1, 2, 3, 4, 5],
            "y_values": [10, 20, 30, 40, 50],
            "project_name": "project",
            "material_name": "material"
        }
        app_module.db = SimpleNamespace(
            spectrums=FakeSpectrumCollection([spec])
        )

        response = app_module.app.test_client().get("/api/experiments")

        self.assertEqual(response.status_code, 200)
        experiment = response.json["experiments"][0]
        self.assertEqual(experiment["id"], str(spec["_id"]))
        self.assertEqual(experiment["point_count"], 5)

    def test_creates_experiment_with_selected_numeric_columns(self):
        app_module.db = SimpleNamespace(
            spectrums=FakeSpectrumCollection([]),
            experiment_groups=FakeExperimentGroupCollection()
        )
        response = app_module.app.test_client().post(
            "/api/experiments",
            data={
                "file": (io.BytesIO(
                    b"temperature,field,permeability,loss\n"
                    b"20,1,2,0.1\n21,2,3,0.2\n22,3,4,0.3\n"
                    b"23,4,5,0.4\n24,5,6,0.5\n"
                ), "measurements.csv"),
                "input_columns": '["temperature", "field"]',
                "target_columns": '["permeability", "loss"]',
                "domain": "MI",
                "experiment_name": "sample experiment"
            },
            content_type="multipart/form-data"
        )

        self.assertEqual(response.status_code, 201)
        experiment = app_module.db.experiment_groups.documents[0]
        self.assertEqual(experiment["input_columns"], ["temperature", "field"])
        self.assertEqual(experiment["target_columns"], ["permeability", "loss"])
        self.assertEqual(experiment["row_count"], 5)
        self.assertEqual(experiment["run_count"], 1)
        self.assertIn(experiment["runs"][0]["csv_file_id"], app_module.fs.files)

        listed = app_module.app.test_client().get("/api/experiments")
        self.assertEqual(listed.status_code, 200)
        listed_experiment = listed.json["experiments"][0]
        self.assertEqual(listed_experiment["experiment_name"], "sample experiment")
        self.assertEqual(listed_experiment["input_columns"], ["temperature", "field"])
        self.assertEqual(listed_experiment["target_columns"], ["permeability", "loss"])
        self.assertEqual(listed_experiment["point_count"], 5)
        self.assertIn("run_id", response.json)

    def test_run_graph_api_loads_a_single_csv(self):
        file_id = ObjectId()
        run_id = "run-one"
        app_module.fs.files[file_id] = (
            b"time_s,current_A,voltage_V\n0,1,0\n1,1,0.5\n"
        )
        group = {
            "_id": ObjectId(),
            "experiment_name": "RC",
            "input_columns": ["time_s", "current_A"],
            "target_columns": ["voltage_V"],
            "runs": [{
                "run_id": run_id,
                "csv_file_id": file_id,
                "csv_filename": "one-amp.csv",
                "row_count": 2
            }]
        }
        app_module.db = SimpleNamespace(
            spectrums=FakeSpectrumCollection([]),
            experiment_groups=FakeExperimentGroupCollection([group])
        )

        response = app_module.app.test_client().get(
            f"/api/experiments/{group['_id']}/runs/{run_id}"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["filename"], "one-amp.csv")
        self.assertEqual(response.json["row_count"], 2)
        self.assertEqual(response.json["input_values"], [[0.0, 1.0], [1.0, 1.0]])
        self.assertEqual(response.json["target_values"], [[0.0], [0.5]])

    def test_data_management_lists_each_run_with_its_own_point_count(self):
        group_id = ObjectId()
        runs = []
        for index in range(2):
            file_id = ObjectId()
            app_module.fs.files[file_id] = b"time_s,voltage_V\n0,0\n1,1\n"
            runs.append({
                "run_id": f"run-{index}",
                "csv_file_id": file_id,
                "csv_filename": f"run-{index}.csv",
                "row_count": 21
            })
        group = {
            "_id": group_id,
            "project_name": "project",
            "material_name": "sample",
            "domain": "EMC",
            "experiment_name": "current",
            "input_columns": ["time_s"],
            "target_columns": ["voltage_V"],
            "row_count": 42,
            "runs": runs
        }
        app_module.db = SimpleNamespace(
            images=FakeSpectrumCollection([]),
            spectrums=FakeSpectrumCollection([]),
            experiment_groups=FakeExperimentGroupCollection([group])
        )
        client = app_module.socketio.test_client(app_module.app)
        self.addCleanup(client.disconnect)

        client.emit("load messages")
        messages = next(
            event["args"][0]
            for event in client.get_received()
            if event["name"] == "load all messages"
        )
        csv_runs = [message for message in messages if message.get("run_id")]

        self.assertEqual([message["row_count"] for message in csv_runs], [21, 21])
        self.assertEqual([message["group_row_count"] for message in csv_runs], [42, 42])

    def test_prediction_pools_multiple_compatible_datasets(self):
        groups = []
        for dataset_index in range(2):
            file_id = ObjectId()
            rows = ["x,y"]
            rows.extend(
                f"{dataset_index * 5 + value},{2 * (dataset_index * 5 + value)}"
                for value in range(5)
            )
            app_module.fs.files[file_id] = "\n".join(rows).encode()
            groups.append({
                "_id": ObjectId(),
                "project_name": "same-project",
                "experiment_name": f"dataset-{dataset_index}",
                "material_name": "sample",
                "domain": "MI",
                "input_columns": ["x"],
                "target_columns": ["y"],
                "row_count": 5,
                "runs": [{
                    "run_id": f"run-{dataset_index}",
                    "csv_file_id": file_id,
                    "csv_filename": f"run-{dataset_index}.csv",
                    "row_count": 5
                }]
            })
        app_module.db = SimpleNamespace(
            spectrums=FakeSpectrumCollection([]),
            experiment_groups=FakeExperimentGroupCollection(groups)
        )
        app_module.model_cache = {"cache_key": None, "gp_model": None, "scaler": None}

        response = app_module.app.test_client().post(
            "/api/predict_dynamic",
            json={
                "experiment_ids": [str(group["_id"]) for group in groups],
                "training_points": 10,
                "conditions": {"x": 4},
                "model_type": "polynomial"
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["total_points"], 10)
        self.assertEqual(response.json["training_run_count"], 2)
        self.assertEqual(
            response.json["experiment_ids"],
            [str(group["_id"]) for group in groups]
        )
        self.assertIsNone(response.json["prediction_interval"])
        self.assertEqual(
            {candidate["model_type"] for candidate in response.json["model_evaluation"]},
            {"gpr", "polynomial", "random_forest"}
        )

    def test_prediction_rejects_mixed_project_datasets(self):
        groups = []
        for project_name in ("project-a", "project-b"):
            groups.append({
                "_id": ObjectId(),
                "project_name": project_name,
                "input_columns": ["x"],
                "target_columns": ["y"],
                "runs": [{
                    "input_values": [[0], [1], [2], [3], [4]],
                    "target_values": [[0], [1], [2], [3], [4]],
                    "row_count": 5
                }]
            })
        app_module.db = SimpleNamespace(
            spectrums=FakeSpectrumCollection([]),
            experiment_groups=FakeExperimentGroupCollection(groups)
        )

        response = app_module.app.test_client().post(
            "/api/predict_dynamic",
            json={
                "experiment_ids": [str(group["_id"]) for group in groups],
                "training_points": 10,
                "conditions": {"x": 2},
                "model_type": "gpr"
            }
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("同じプロジェクト", response.json["error"])

    def test_deleting_a_run_preserves_other_runs_in_its_training_set(self):
        group_id = ObjectId()
        file_ids = [ObjectId(), ObjectId()]
        runs = [{
            "run_id": f"run-{index}",
            "csv_file_id": file_id,
            "row_count": 21
        } for index, file_id in enumerate(file_ids)]
        for file_id in file_ids:
            app_module.fs.files[file_id] = b"time_s,voltage_V\n0,0\n"
        group = {
            "_id": group_id,
            "row_count": 42,
            "run_count": 2,
            "runs": runs
        }
        app_module.db = SimpleNamespace(
            images=FakeSpectrumCollection([]),
            spectrums=FakeSpectrumCollection([]),
            experiment_groups=FakeExperimentGroupCollection([group])
        )
        client = app_module.socketio.test_client(app_module.app)
        self.addCleanup(client.disconnect)

        client.emit(
            "delete message",
            {"id": str(group_id), "run_id": "run-0"}
        )

        self.assertEqual(group["row_count"], 21)
        self.assertEqual(group["run_count"], 1)
        self.assertEqual(group["runs"][0]["run_id"], "run-1")
        self.assertNotIn(file_ids[0], app_module.fs.files)
        self.assertIn(file_ids[1], app_module.fs.files)

    def test_combines_separate_runs_and_predicts_an_unmeasured_input(self):
        app_module.db = SimpleNamespace(
            spectrums=FakeSpectrumCollection([]),
            experiment_groups=FakeExperimentGroupCollection()
        )
        client = app_module.app.test_client()

        def upload_run(current, filename):
            csv_data = (
                "current,time,voltage\n"
                f"{current},0,0\n{current},1,1\n{current},2,2\n"
            ).encode()
            return client.post(
                "/api/experiments",
                data={
                    "file": (io.BytesIO(csv_data), filename),
                    "project_name": "circuits",
                    "material_name": "RC",
                    "domain": "EMC",
                    "experiment_name": "RC response",
                    "input_columns": '["current", "time"]',
                    "target_columns": '["voltage"]'
                },
                content_type="multipart/form-data"
            )

        first = upload_run(1, "one-amp.csv")
        second = upload_run(3, "three-amp.csv")
        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 201)
        self.assertEqual(first.json["id"], second.json["id"])
        self.assertEqual(second.json["row_count"], 6)
        self.assertEqual(second.json["run_count"], 2)

        group = app_module.db.experiment_groups.documents[0]
        input_columns, target_columns, input_values, _ = app_module.load_experiment_table(group)
        self.assertEqual(input_columns, ["current", "time"])
        self.assertEqual(target_columns, ["voltage"])
        self.assertEqual(input_values.shape, (6, 2))
        self.assertEqual(set(input_values[:, 0]), {1, 3})

        app_module.model_cache = {
            "cache_key": None,
            "gp_model": None,
            "scaler": None
        }

        class FakeRegressor:
            def __init__(self, **kwargs):
                pass

            def fit(self, x_values, y_values):
                return self

            def predict(self, x_values, return_std):
                return np.zeros((len(x_values), 1)), np.ones((len(x_values), 1))

        original_regressor = app_module.GaussianProcessRegressor
        app_module.GaussianProcessRegressor = FakeRegressor
        self.addCleanup(
            setattr,
            app_module,
            "GaussianProcessRegressor",
            original_regressor
        )

        response = client.post(
            "/api/predict_dynamic",
            json={
                "experiment_id": first.json["id"],
                "training_points": 5,
                "conditions": {"current": 5, "time": 1}
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["input_conditions"]["current"], 5)
        self.assertEqual(response.json["training_ranges"]["current"], [1, 3])
        self.assertEqual(response.json["out_of_range_conditions"], {"current": 5})

    def test_gpr_posterior_uncertainty_grows_outside_training_region(self):
        input_values = np.linspace(0, 5, 21).reshape(-1, 1)
        target_values = np.sin(input_values)
        model = app_module.fit_experiment_candidate(
            "gpr",
            input_values,
            target_values,
            ["time_s"],
            ["voltage_V"],
            optimize_gpr=False
        )
        _, standard_deviation = app_module.predict_experiment_candidate(
            model,
            np.array([[2.5], [15.0]])
        )

        self.assertIsNotNone(standard_deviation)
        self.assertGreater(standard_deviation[1, 0], standard_deviation[0, 0])

    def test_model_comparison_uses_whole_run_validation(self):
        times = np.tile(np.arange(21, dtype=float) * 0.25, 2)
        currents = np.repeat([1.0, 3.0], 21)
        input_values = np.column_stack((currents, times))
        target_values = (
            currents * 1.5 * (1 - np.exp(-times / 2))
        ).reshape(-1, 1)
        spec = {
            "_id": "run-validation",
            "runs": [
                {"csv_file_id": "one-amp", "row_count": 21},
                {"csv_file_id": "three-amp", "row_count": 21}
            ]
        }
        app_module.model_validation_cache = {"cache_key": None, "result": None}

        validation = app_module.validate_experiment_models(
            spec,
            ["current_A", "time_s"],
            ["voltage_V"],
            input_values,
            target_values,
            42
        )

        self.assertTrue(validation["available"])
        self.assertEqual(validation["fold_count"], 2)
        self.assertEqual(
            {candidate["model_type"] for candidate in validation["candidates"]},
            {"gpr", "polynomial", "random_forest"}
        )
        self.assertTrue(all(
            np.isfinite(candidate["normalized_rmse"])
            for candidate in validation["candidates"]
        ))
        app_module.model_cache = {
            "cache_key": None,
            "gp_model": None,
            "scaler": None
        }
        selected_model, reason, model, _ = app_module.prepare_experiment_model(
            spec,
            ["current_A", "time_s"],
            ["voltage_V"],
            input_values,
            target_values,
            42,
            "auto"
        )
        prediction, _ = app_module.predict_experiment_candidate(
            model, np.array([[5.0, 15.0]])
        )
        self.assertIn(selected_model, {"gpr", "polynomial", "random_forest"})
        self.assertEqual(reason, "run_holdout")
        self.assertTrue(np.isfinite(prediction[0, 0]))

    def test_prediction_grid_returns_gpr_interval_and_measured_runs(self):
        run_files = [ObjectId(), ObjectId()]
        runs = []
        for current, file_id in zip([1.0, 3.0], run_files):
            rows = ["current_A,time_s,voltage_V"]
            for index in range(21):
                time = index * 0.25
                voltage = current * 1.5 * (1 - np.exp(-time / 2))
                rows.append(f"{current:g},{time:.2f},{voltage:.4f}")
            app_module.fs.files[file_id] = "\n".join(rows).encode()
            runs.append({
                "csv_file_id": file_id,
                "row_count": 21
            })

        spec = {
            "_id": ObjectId(),
            "input_columns": ["current_A", "time_s"],
            "target_columns": ["voltage_V"],
            "row_count": 42,
            "runs": runs
        }
        app_module.db = SimpleNamespace(
            spectrums=FakeSpectrumCollection([]),
            experiment_groups=FakeExperimentGroupCollection([spec])
        )
        app_module.model_cache = {
            "cache_key": None,
            "gp_model": None,
            "scaler": None
        }
        app_module.model_validation_cache = {"cache_key": None, "result": None}

        response = app_module.app.test_client().post(
            "/api/predict_grid",
            json={
                "experiment_id": str(spec["_id"]),
                "training_points": 42,
                "conditions": {"current_A": 5, "time_s": 0},
                "x_column": "time_s",
                "y_column": None,
                "x_range": [0, 15],
                "grid_size": 5,
                "model_type": "gpr"
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["model_type"], "gpr")
        self.assertEqual(
            response.json["prediction_interval"]["method"],
            "gaussian_process_posterior_predictive"
        )
        self.assertEqual(response.json["validation_runs"], 2)
        self.assertEqual(response.json["out_of_range_conditions"], {"current_A": 5})
        self.assertEqual(response.json["out_of_range_axes"], {"time_s": [0, 15]})
        self.assertEqual(len(response.json["measured_data"]), 2)
        self.assertEqual(
            sum(len(run["x_values"]) for run in response.json["measured_data"]),
            42
        )

    def test_imports_more_than_800_rows_but_caps_gpr_training_sample(self):
        csv_lines = ["x,y"]
        csv_lines.extend(f"{value},{value * 2}" for value in range(1200))
        app_module.db = SimpleNamespace(
            spectrums=FakeSpectrumCollection([]),
            experiment_groups=FakeExperimentGroupCollection()
        )
        response = app_module.app.test_client().post(
            "/api/experiments",
            data={
                "file": (io.BytesIO("\n".join(csv_lines).encode()), "large.csv"),
                "input_columns": '["x"]',
                "target_columns": '["y"]',
                "experiment_name": "large"
            },
            content_type="multipart/form-data"
        )

        self.assertEqual(response.status_code, 201)
        spec = app_module.db.experiment_groups.documents[0]
        self.assertEqual(spec["row_count"], 1200)

        invalid_training_request = app_module.app.test_client().post(
            "/api/predict_dynamic",
            json={
                "experiment_id": str(spec["_id"]),
                "training_points": 1200,
                "conditions": {"x": 500}
            }
        )
        self.assertEqual(invalid_training_request.status_code, 400)
        self.assertIn("1000", invalid_training_request.json["error"])

        invalid_grid_request = app_module.app.test_client().post(
            "/api/predict_grid",
            json={
                "experiment_id": str(spec["_id"]),
                "training_points": 1200,
                "conditions": {"x": 500},
                "x_column": "x",
                "y_column": None,
                "grid_size": 5
            }
        )
        self.assertEqual(invalid_grid_request.status_code, 400)
        self.assertIn("1000", invalid_grid_request.json["error"])

    def test_rejects_non_string_column_names(self):
        with self.assertRaisesRegex(ValueError, "列名の指定"):
            app_module.parse_numeric_columns(
                ["x", "y"], [["1", "2"]] * 5, [[],], ["y"]
            )

    def test_csv_preview_returns_headers_and_sample_rows(self):
        response = app_module.app.test_client().post(
            "/api/csv/preview",
            data={
                "file": (io.BytesIO(
                    b"temperature,field,loss\n"
                    b"20,1,0.1\n21,2,0.2\n22,3,0.3\n"
                    b"23,4,0.4\n24,5,0.5\n"
                ), "measurements.csv")
            },
            content_type="multipart/form-data"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["columns"], ["temperature", "field", "loss"])
        self.assertEqual(response.json["preview"][0], ["20", "1", "0.1"])
        self.assertEqual(response.json["row_count"], 5)

    def test_prediction_uses_selected_experiment_and_even_sample(self):
        spec = {
            "_id": ObjectId(),
            "x_values": list(range(9)),
            "y_values": [value * 10 for value in range(9)],
            "project_name": "project",
            "material_name": "material"
        }
        app_module.db = SimpleNamespace(
            spectrums=FakeSpectrumCollection([spec])
        )
        app_module.save_experiment_csv(
            spec["_id"], spec["x_values"], spec["y_values"]
        )
        app_module.model_cache = {"cache_key": None, "gp_model": None}

        class FakeRegressor:
            fitted_x = None

            def __init__(self, **kwargs):
                pass

            def fit(self, x_values, y_values):
                FakeRegressor.fitted_x = x_values[:, 0].tolist()
                return self

            def predict(self, x_values, return_std):
                return np.zeros((len(x_values), 1)), np.ones((len(x_values), 1))

        original_regressor = app_module.GaussianProcessRegressor
        app_module.GaussianProcessRegressor = FakeRegressor
        self.addCleanup(
            setattr,
            app_module,
            "GaussianProcessRegressor",
            original_regressor
        )

        response = app_module.app.test_client().post(
            "/api/predict_dynamic",
            json={
                "experiment_id": str(spec["_id"]),
                "training_points": 5,
                "conditions": {"X": 4}
            }
        )

        self.assertEqual(response.status_code, 200)
        np.testing.assert_allclose(
            FakeRegressor.fitted_x,
            [-np.sqrt(2), -np.sqrt(0.5), 0, np.sqrt(0.5), np.sqrt(2)]
        )
        self.assertEqual(response.json["training_points"], 5)
        self.assertEqual(response.json["experiment_id"], str(spec["_id"]))
        self.assertEqual(response.json["predictions"], {"Y": 0.0})

    def test_prediction_returns_multiple_results_for_dynamic_conditions(self):
        csv_bytes = (
            b"temperature,field,permeability,loss\n"
            b"20,1,2,0.1\n21,2,3,0.2\n22,3,4,0.3\n"
            b"23,4,5,0.4\n24,5,6,0.5\n"
        )
        file_id = ObjectId()
        app_module.fs.files[file_id] = csv_bytes
        spec = {
            "_id": ObjectId(),
            "csv_file_id": file_id,
            "input_columns": ["temperature", "field"],
            "target_columns": ["permeability", "loss"],
            "row_count": 5
        }
        app_module.db = SimpleNamespace(
            spectrums=FakeSpectrumCollection([spec])
        )
        app_module.model_cache = {"cache_key": None, "gp_model": None, "scaler": None}

        class FakeRegressor:
            def __init__(self, **kwargs):
                pass

            def fit(self, x_values, y_values):
                self.target_count = y_values.shape[1]
                return self

            def predict(self, x_values, return_std):
                return (
                    np.array([[10.0, 20.0]]),
                    np.array([[1.0, 2.0]])
                )

        original_regressor = app_module.GaussianProcessRegressor
        app_module.GaussianProcessRegressor = FakeRegressor
        self.addCleanup(
            setattr,
            app_module,
            "GaussianProcessRegressor",
            original_regressor
        )

        response = app_module.app.test_client().post(
            "/api/predict_dynamic",
            json={
                "experiment_id": str(spec["_id"]),
                "training_points": 5,
                "conditions": {"temperature": 22, "field": 3}
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["predictions"], {
            "permeability": 10.0,
            "loss": 20.0
        })
        self.assertEqual(response.json["confidence_low"], {
            "permeability": 8.04,
            "loss": 16.08
        })

    def test_prediction_grid_supports_surfaces_and_single_axis_bars(self):
        csv_bytes = (
            b"wavelength,time,intensity,absorbance\n"
            b"400,0,1,0.1\n450,1,2,0.2\n500,2,3,0.3\n"
            b"550,3,4,0.4\n600,4,5,0.5\n"
        )
        file_id = ObjectId()
        app_module.fs.files[file_id] = csv_bytes
        spec = {
            "_id": ObjectId(),
            "csv_file_id": file_id,
            "input_columns": ["wavelength", "time"],
            "target_columns": ["intensity", "absorbance"],
            "row_count": 5
        }
        app_module.db = SimpleNamespace(
            spectrums=FakeSpectrumCollection([spec])
        )
        app_module.model_cache = {
            "cache_key": None,
            "gp_model": None,
            "scaler": None
        }

        class FakeRegressor:
            def __init__(self, **kwargs):
                pass

            def fit(self, x_values, y_values):
                return self

            def predict(self, x_values, return_std):
                return (
                    np.column_stack((
                        x_values[:, 0] + x_values[:, 1],
                        x_values[:, 0] - x_values[:, 1]
                    )),
                    np.ones((len(x_values), 2))
                )

        original_regressor = app_module.GaussianProcessRegressor
        app_module.GaussianProcessRegressor = FakeRegressor
        self.addCleanup(
            setattr,
            app_module,
            "GaussianProcessRegressor",
            original_regressor
        )
        client = app_module.app.test_client()
        conditions = {"wavelength": 500, "time": 2}

        surface = client.post(
            "/api/predict_grid",
            json={
                "experiment_id": str(spec["_id"]),
                "training_points": 5,
                "conditions": conditions,
                "x_column": "wavelength",
                "y_column": "time",
                "x_range": [350, 650],
                "y_range": [-1, 5],
                "grid_size": 5
            }
        )
        self.assertEqual(surface.status_code, 200)
        self.assertEqual(surface.json["x_range"], [350, 650])
        self.assertEqual(surface.json["y_range"], [-1, 5])
        self.assertEqual(
            surface.json["out_of_range_axes"],
            {"wavelength": [350, 650], "time": [-1, 5]}
        )
        self.assertEqual(len(surface.json["x_values"]), 5)
        self.assertEqual(len(surface.json["y_values"]), 5)
        self.assertEqual(len(surface.json["predictions"]["intensity"]), 5)
        self.assertEqual(len(surface.json["predictions"]["intensity"][0]), 5)
        self.assertEqual(len(surface.json["predictions"]["absorbance"][0]), 5)
        self.assertEqual(len(surface.json["confidence_low"]["intensity"]), 5)
        self.assertEqual(len(surface.json["confidence_low"]["intensity"][0]), 5)

        line = client.post(
            "/api/predict_grid",
            json={
                "experiment_id": str(spec["_id"]),
                "training_points": 5,
                "conditions": conditions,
                "x_column": "wavelength",
                "y_column": None,
                "target_column": "intensity",
                "compare_models": True,
                "grid_size": 5
            }
        )
        self.assertEqual(line.status_code, 200)
        self.assertIsNone(line.json["y_values"])
        self.assertEqual(len(line.json["predictions"]["intensity"]), 5)
        self.assertEqual(
            set(line.json["model_comparison_predictions"]),
            {"gpr", "polynomial", "random_forest"}
        )
        self.assertEqual(
            len(line.json["model_comparison_predictions"]["polynomial"]["intensity"]),
            5
        )


if __name__ == "__main__":
    unittest.main()
