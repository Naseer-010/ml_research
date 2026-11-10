"""Unit tests for dataset cleaning and standardization routines in data_loader."""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    _clean_credit_default,
    _clean_german_credit,
    _clean_bank_marketing,
    _clean_adult,
    CREDIT_DEFAULT_COLUMN_MAP,
)


class TestCreditDefaultCleaner(unittest.TestCase):
    """Test cleaning and column mapping for Credit Default dataset."""

    def test_clean_credit_default_with_x_codes(self):
        """UCI ID 350 features come with X1..X23 codes; verify they are mapped to standard names."""
        raw_cols = [f"X{i}" for i in range(1, 24)] + ["default.payment"]
        sample_data = {col: [1, 2, 1] for col in raw_cols}
        sample_data["X2"] = [1, 2, 1]  # SEX: 1=Male, 2=Female
        sample_data["X3"] = [1, 2, 3]  # EDUCATION: 1=Grad School, 2=University, 3=High School
        sample_data["X4"] = [1, 2, 3]  # MARRIAGE: 1=Married, 2=Single, 3=Other
        sample_data["default.payment"] = [0, 1, 0]

        df_raw = pd.DataFrame(sample_data)
        df_clean = _clean_credit_default(df_raw)

        # Check sensitive columns exist
        self.assertIn("SEX", df_clean.columns)
        self.assertIn("EDUCATION", df_clean.columns)
        self.assertIn("MARRIAGE", df_clean.columns)
        self.assertIn("LIMIT_BAL", df_clean.columns)
        self.assertIn("default.payment", df_clean.columns)

        # Check demographic mappings
        self.assertListEqual(df_clean["SEX"].tolist(), ["Male", "Female", "Male"])
        self.assertListEqual(df_clean["EDUCATION"].tolist(), ["Grad School", "University", "High School"])
        self.assertListEqual(df_clean["MARRIAGE"].tolist(), ["Married", "Single", "Other"])
        self.assertListEqual(df_clean["default.payment"].tolist(), [0, 1, 0])

    def test_clean_credit_default_already_named(self):
        """Verify cleaner works when columns already have descriptive names."""
        data = {
            "LIMIT_BAL": [20000, 50000],
            "SEX": ["Male", "Female"],
            "EDUCATION": ["University", "Grad School"],
            "MARRIAGE": ["Single", "Married"],
            "AGE": [24, 30],
            "default.payment": [1, 0],
        }
        df_raw = pd.DataFrame(data)
        df_clean = _clean_credit_default(df_raw)

        self.assertIn("SEX", df_clean.columns)
        self.assertIn("EDUCATION", df_clean.columns)
        self.assertIn("MARRIAGE", df_clean.columns)
        self.assertListEqual(df_clean["default.payment"].tolist(), [1, 0])

    def test_clean_credit_default_drops_id(self):
        """Verify client ID column is dropped if present."""
        raw_cols = ["ID"] + [f"X{i}" for i in range(1, 24)] + ["default.payment"]
        sample_data = {col: [1] for col in raw_cols}
        df_raw = pd.DataFrame(sample_data)
        df_clean = _clean_credit_default(df_raw)

        self.assertNotIn("ID", df_clean.columns)

    def test_clean_credit_default_missing_sensitive_raises(self):
        """Missing sensitive attribute after cleaning must raise KeyError."""
        df_bad = pd.DataFrame({"X1": [100], "default.payment": [0]})
        with self.assertRaises(KeyError):
            _clean_credit_default(df_bad)


if __name__ == "__main__":
    unittest.main()

