"""Unit tests for preprocessing and column type identification."""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import identify_column_types, preprocess_dataset


class TestPreprocessing(unittest.TestCase):
    """Test column identification and dataset preprocessing across data types."""

    def test_identify_column_types_with_string_dtype(self):
        """Verify identify_column_types correctly handles pandas StringDtype without error."""
        df = pd.DataFrame({
            "num1": [1.0, 2.0, 3.0],
            "num2": [10, 20, 30],
            "str_obj": ["a", "b", "c"],
            "str_dtype": pd.Series(["x", "y", "z"], dtype="string"),
            "cat_col": pd.Series(["cat1", "cat2", "cat1"], dtype="category"),
            "sens_num": [1, 2, 1],
        })

        num_cols, cat_cols = identify_column_types(df, sensitive_columns=["sens_num"])

        self.assertIn("num1", num_cols)
        self.assertIn("num2", num_cols)
        self.assertIn("str_obj", cat_cols)
        self.assertIn("str_dtype", cat_cols)
        self.assertIn("cat_col", cat_cols)
        self.assertIn("sens_num", cat_cols)  # Sensitive with <= 20 unique levels treated as categorical

    def test_preprocess_dataset_smoke(self):
        """Smoke test full preprocess_dataset on a synthetic Adult-like dataframe."""
        df = pd.DataFrame({
            "age": [25, 45, 30, 50, 35, 60] * 10,
            "workclass": ["Private", "Self-emp", "State-gov", "Private", "Private", "Federal-gov"] * 10,
            "sex": ["Male", "Female", "Male", "Female", "Male", "Female"] * 10,
            "race": ["White", "Black", "White", "Asian", "White", "Other"] * 10,
            "income": [0, 1, 0, 1, 0, 1] * 10,
        })
        # Convert workclass to string dtype to test pandas 3.0 compatibility
        df["workclass"] = df["workclass"].astype("string")

        prepared = preprocess_dataset(df, dataset_key="adult", seed=42)

        self.assertEqual(prepared.X_train.shape[0], 36)
        self.assertEqual(prepared.X_cal.shape[0], 12)
        self.assertEqual(prepared.X_test.shape[0], 12)
        self.assertIn("sex", prepared.groups_test)
        self.assertIn("race", prepared.groups_test)
        self.assertEqual(prepared.X_train.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()

