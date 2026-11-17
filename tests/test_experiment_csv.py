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


class ExperimentCsvTests(unittest.TestCase):
    def setUp(self):
        self.original_instance_path = app_module.app.instance_path
        self.original_db = app_module.db
        self.original_model_cache = app_module.model_cache.copy()
        self.temp_directory = tempfile.TemporaryDirectory()
        app_module.app.instance_path = self.temp_directory.name

    def tearDown(self):
        app_module.app.instance_path = self.original_instance_path
        app_module.db = self.original_db
        app_module.model_cache = self.original_model_cache
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
                return np.zeros(len(x_values)), np.ones(len(x_values))

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
                "training_points": 5
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(FakeRegressor.fitted_x, [0, 2, 4, 6, 8])
        self.assertEqual(response.json["training_points"], 5)
        self.assertEqual(response.json["experiment_id"], str(spec["_id"]))


if __name__ == "__main__":
    unittest.main()
