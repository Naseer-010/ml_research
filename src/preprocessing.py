"""Preprocessing, feature engineering, and stratified splitting module.

Prepares tabular datasets for model training, post-hoc calibration, and
group-wise calibration evaluation.

Key functionality:
- 3-way stratified splitting into Train (60%), Calibration (20%), and Test (20%).
- Extraction of sensitive group labels prior to feature encoding.
- Retaining sensitive columns as features for model inputs while tracking them
  separately for downstream fairness/calibration evaluation.
- Categorical encoding with OrdinalEncoder (handling unseen categories gracefully).
- Missing value imputation (median for numeric, mode for categorical).
- Standardization of continuous features with StandardScaler.
- Fitting all transformers strictly on the training partition to prevent data leakage.
- Returning a structured PreparedData container that satisfies both dictionary and
  object/dataclass interfaces.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

logger = logging.getLogger(__name__)
if not logger.handlers and not logging.getLogger().handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(config.LOG_FORMAT, datefmt=getattr(config, "LOG_DATEFMT", "%H:%M:%S"))
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))


class PreparedData(dict):
    """Container for preprocessed dataset partitions, labels, and group attributes.

    Inherits from dict so it can be indexed like a dictionary (e.g. data['X_train'])
    while also providing typed property access (e.g. data.X_train) and dataclass-like
    convenience.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        X_cal: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_cal: np.ndarray,
        y_test: np.ndarray,
        groups_train: Dict[str, np.ndarray],
        groups_cal: Dict[str, np.ndarray],
        groups_test: Dict[str, np.ndarray],
        feature_names: List[str],
        sensitive_columns: List[str],
        preprocessor: Optional[ColumnTransformer] = None,
    ) -> None:
        """Initialize the PreparedData container.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix (float32).
        X_cal : np.ndarray
            Calibration feature matrix (float32).
        X_test : np.ndarray
            Test feature matrix (float32).
        y_train : np.ndarray
            Binary training labels (int64).
        y_cal : np.ndarray
            Binary calibration labels (int64).
        y_test : np.ndarray
            Binary test labels (int64).
        groups_train : Dict[str, np.ndarray]
            Mapping of sensitive attribute names to training group label arrays.
        groups_cal : Dict[str, np.ndarray]
            Mapping of sensitive attribute names to calibration group label arrays.
        groups_test : Dict[str, np.ndarray]
            Mapping of sensitive attribute names to test group label arrays.
        feature_names : List[str]
            List of feature names matching the columns of the X matrices.
        sensitive_columns : List[str]
            List of sensitive column names extracted for calibration evaluation.
        preprocessor : Optional[ColumnTransformer], default=None
            Fitted ColumnTransformer used for feature preprocessing.
        """
        super().__init__(
            X_train=X_train,
            X_cal=X_cal,
            X_test=X_test,
            y_train=y_train,
            y_cal=y_cal,
            y_test=y_test,
            groups_train=groups_train,
            groups_cal=groups_cal,
            groups_test=groups_test,
            feature_names=feature_names,
            sensitive_columns=sensitive_columns,
        )
        self.preprocessor = preprocessor

    @property
    def X_train(self) -> np.ndarray:
        """Training feature array."""
        return self["X_train"]

    @property
    def X_cal(self) -> np.ndarray:
        """Calibration feature array."""
        return self["X_cal"]

    @property
    def X_test(self) -> np.ndarray:
        """Test feature array."""
        return self["X_test"]

    @property
    def y_train(self) -> np.ndarray:
        """Training binary targets."""
        return self["y_train"]

    @property
    def y_cal(self) -> np.ndarray:
        """Calibration binary targets."""
        return self["y_cal"]

    @property
    def y_test(self) -> np.ndarray:
        """Test binary targets."""
        return self["y_test"]

    @property
    def groups_train(self) -> Dict[str, np.ndarray]:
        """Training sensitive group arrays."""
        return self["groups_train"]

    @property
    def groups_cal(self) -> Dict[str, np.ndarray]:
        """Calibration sensitive group arrays."""
        return self["groups_cal"]

    @property
    def groups_test(self) -> Dict[str, np.ndarray]:
        """Test sensitive group arrays."""
        return self["groups_test"]

    @property
    def feature_names(self) -> List[str]:
        """Names of encoded features in column order."""
        return self["feature_names"]

    @property
    def sensitive_columns(self) -> List[str]:
        """Names of sensitive attributes."""
        return self["sensitive_columns"]

    def to_dict(self) -> Dict[str, Any]:
        """Return a copy of the dictionary representation.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all split data and metadata.
        """
        return dict(self)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        preprocessor: Optional[ColumnTransformer] = None,
    ) -> "PreparedData":
        """Instantiate a PreparedData container from a dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing the expected keys.
        preprocessor : Optional[ColumnTransformer], default=None
            Optional fitted ColumnTransformer.

        Returns
        -------
        PreparedData
            Constructed PreparedData instance.
        """
        return cls(
            X_train=data["X_train"],
            X_cal=data["X_cal"],
            X_test=data["X_test"],
            y_train=data["y_train"],
            y_cal=data["y_cal"],
            y_test=data["y_test"],
            groups_train=data["groups_train"],
            groups_cal=data["groups_cal"],
            groups_test=data["groups_test"],
            feature_names=data["feature_names"],
            sensitive_columns=data["sensitive_columns"],
            preprocessor=preprocessor or data.get("preprocessor"),
        )


def split_data(
    df: pd.DataFrame,
    target_column: str,
    train_ratio: float = config.TRAIN_RATIO,
    cal_ratio: float = config.CALIBRATION_RATIO,
    test_ratio: float = config.TEST_RATIO,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform a 3-way stratified split into train, calibration, and test partitions.

    Splits df such that the target class distribution is preserved across all
    three splits according to train_ratio, cal_ratio, and test_ratio.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset DataFrame.
    target_column : str
        Column name to stratify splits on.
    train_ratio : float, default=0.60
        Fraction of samples for the training set.
    cal_ratio : float, default=0.20
        Fraction of samples for post-hoc calibration set.
    test_ratio : float, default=0.20
        Fraction of samples for the final evaluation test set.
    seed : int, default=42
        Random seed for reproducible partitioning.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Tuple of (df_train, df_cal, df_test).
    """
    total_ratio = train_ratio + cal_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0; received {train_ratio} + {cal_ratio} + {test_ratio} = {total_ratio}"
        )

    # First split off the test set
    try:
        df_train_cal, df_test = train_test_split(
            df,
            test_size=test_ratio,
            random_state=seed,
            stratify=df[target_column],
        )
    except ValueError as exc:
        logger.warning(
            "Stratified split for test set failed: %s. Falling back to unstratified split.",
            exc,
        )
        df_train_cal, df_test = train_test_split(
            df,
            test_size=test_ratio,
            random_state=seed,
        )

    # Next split train_cal into train and calibration sets
    # Relative ratio for calibration within the remaining data
    relative_cal_ratio = cal_ratio / (train_ratio + cal_ratio)

    try:
        df_train, df_cal = train_test_split(
            df_train_cal,
            test_size=relative_cal_ratio,
            random_state=seed,
            stratify=df_train_cal[target_column],
        )
    except ValueError as exc:
        logger.warning(
            "Stratified split for calibration set failed: %s. Falling back to unstratified split.",
            exc,
        )
        df_train, df_cal = train_test_split(
            df_train_cal,
            test_size=relative_cal_ratio,
            random_state=seed,
        )

    df_train = df_train.reset_index(drop=True)
    df_cal = df_cal.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    logger.info(
        "Split sizes — Train: %d (%.1f%%) | Cal: %d (%.1f%%) | Test: %d (%.1f%%)",
        len(df_train),
        (len(df_train) / len(df)) * 100,
        len(df_cal),
        (len(df_cal) / len(df)) * 100,
        len(df_test),
        (len(df_test) / len(df)) * 100,
    )
    return df_train, df_cal, df_test


def identify_column_types(
    df: pd.DataFrame,
    sensitive_columns: List[str],
) -> Tuple[List[str], List[str]]:
    """Identify numerical and categorical column names in feature DataFrame.

    Columns with string, object, category, or boolean data types, as well
    as discrete sensitive attributes, are treated as categorical. Continuous
    numeric features are treated as numerical.

    Parameters
    ----------
    df : pd.DataFrame
        Features DataFrame (target excluded).
    sensitive_columns : List[str]
        List of sensitive attribute column names.

    Returns
    -------
    Tuple[List[str], List[str]]
        Tuple of (numerical_cols, categorical_cols).
    """
    numerical_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in df.columns:
        dtype = df[col].dtype
        is_categorical = (
            isinstance(dtype, (pd.CategoricalDtype, pd.StringDtype))
            or pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or dtype.name in ["object", "string", "category", "bool", "str"]
            or "string" in str(dtype).lower()
        )

        if is_categorical:
            categorical_cols.append(col)
        elif col in sensitive_columns and df[col].nunique() <= 20:
            # Sensitive columns with discrete levels are treated as categorical
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)

    return numerical_cols, categorical_cols


def build_preprocessor(
    numerical_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer with median/mode imputation and scaling.

    Numerical features:
        SimpleImputer(strategy='median') -> StandardScaler()
    Categorical features:
        SimpleImputer(strategy='most_frequent') -> OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    Parameters
    ----------
    numerical_cols : List[str]
        List of numeric column names.
    categorical_cols : List[str]
        List of categorical column names.

    Returns
    -------
    ColumnTransformer
        Unfitted scikit-learn ColumnTransformer.
    """
    transformers = []

    if numerical_cols:
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipeline, numerical_cols))

    if categorical_cols:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        transformers.append(("cat", cat_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def preprocess_dataset(
    df: pd.DataFrame,
    dataset_key: str,
    seed: int,
    drop_sensitive_features: bool = False,
) -> PreparedData:
    """Perform stratified splitting, sensitive attribute extraction, and preprocessing.

    1. Stratified 3-way split on binary target (train: 60%, cal: 20%, test: 20%).
    2. Sensitive group extraction BEFORE feature encoding.
    3. Retains sensitive attributes in feature matrix (unless drop_sensitive_features=True).
    4. Imputation and encoding:
       - Numerical: median imputation + StandardScaler
       - Categorical: mode imputation + OrdinalEncoder
    5. Fitting all transformers strictly on train partition.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset DataFrame containing target and sensitive columns.
    dataset_key : str
        Dataset key configured in config.DATASETS.
    seed : int
        Random seed for reproducible splitting and preprocessing.
    drop_sensitive_features : bool, default=False
        If True, drops sensitive attributes from input features X.
        By default False, keeping them as features for the models while also
        recording them separately for group-wise evaluation.

    Returns
    -------
    PreparedData
        Structured container (also behaves as dict) containing X, y, groups,
        feature_names, sensitive_columns, and the fitted preprocessor.
    """
    if dataset_key not in config.DATASETS:
        raise ValueError(
            f"Unknown dataset key '{dataset_key}'. Expected one of {list(config.DATASETS.keys())}"
        )

    cfg = config.DATASETS[dataset_key]
    target_col = cfg.target_column
    sensitive_cols = [c for c in cfg.sensitive_attributes if c in df.columns]

    if not sensitive_cols:
        logger.warning(
            "None of the expected sensitive attributes %s were found in DataFrame columns: %s",
            cfg.sensitive_attributes,
            list(df.columns),
        )

    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in DataFrame for dataset '{dataset_key}'."
        )

    # 1. Stratified 3-way partition
    df_train, df_cal, df_test = split_data(
        df=df,
        target_column=target_col,
        train_ratio=config.TRAIN_RATIO,
        cal_ratio=config.CALIBRATION_RATIO,
        test_ratio=config.TEST_RATIO,
        seed=seed,
    )

    # 2. Extract sensitive group labels BEFORE any transformation
    groups_train: Dict[str, np.ndarray] = {
        col: df_train[col].astype(str).to_numpy() for col in sensitive_cols
    }
    groups_cal: Dict[str, np.ndarray] = {
        col: df_cal[col].astype(str).to_numpy() for col in sensitive_cols
    }
    groups_test: Dict[str, np.ndarray] = {
        col: df_test[col].astype(str).to_numpy() for col in sensitive_cols
    }

    # 3. Extract binary targets
    y_train = df_train[target_col].to_numpy(dtype=np.int64)
    y_cal = df_cal[target_col].to_numpy(dtype=np.int64)
    y_test = df_test[target_col].to_numpy(dtype=np.int64)

    # 4. Separate feature matrices
    drop_cols = [target_col]
    if drop_sensitive_features:
        drop_cols.extend(sensitive_cols)
        logger.info(
            "Dropping sensitive columns from features: %s", sensitive_cols
        )
    else:
        logger.info(
            "Retaining sensitive columns as features: %s", sensitive_cols
        )

    X_train_df = df_train.drop(columns=drop_cols).copy()
    X_cal_df = df_cal.drop(columns=drop_cols).copy()
    X_test_df = df_test.drop(columns=drop_cols).copy()

    # 5. Identify column types on training features
    num_cols, cat_cols = identify_column_types(X_train_df, sensitive_cols)
    logger.info(
        "Identified %d numerical features and %d categorical features.",
        len(num_cols),
        len(cat_cols),
    )

    # 6. Build and fit preprocessor strictly on train set
    preprocessor = build_preprocessor(num_cols, cat_cols)
    preprocessor.fit(X_train_df)

    # 7. Transform train, calibration, and test sets
    def _to_dense_f32(arr: Any) -> np.ndarray:
        if hasattr(arr, "toarray"):
            arr = arr.toarray()
        return np.asarray(arr, dtype=np.float32)

    X_train = _to_dense_f32(preprocessor.transform(X_train_df))
    X_cal = _to_dense_f32(preprocessor.transform(X_cal_df))
    X_test = _to_dense_f32(preprocessor.transform(X_test_df))

    # 8. Feature names matching the order of ColumnTransformer
    feature_names = list(num_cols) + list(cat_cols)

    logger.info(
        "Preprocessing complete for '%s' (seed=%d): X_train=%s, X_cal=%s, X_test=%s",
        dataset_key,
        seed,
        str(X_train.shape),
        str(X_cal.shape),
        str(X_test.shape),
    )

    return PreparedData(
        X_train=X_train,
        X_cal=X_cal,
        X_test=X_test,
        y_train=y_train,
        y_cal=y_cal,
        y_test=y_test,
        groups_train=groups_train,
        groups_cal=groups_cal,
        groups_test=groups_test,
        feature_names=feature_names,
        sensitive_columns=sensitive_cols,
        preprocessor=preprocessor,
    )


def get_preprocessed_dataset(
    dataset_key: str,
    seed: int = 42,
    force_download: bool = False,
    drop_sensitive_features: bool = False,
) -> PreparedData:
    """High-level utility to load and preprocess a dataset in a single call.

    Parameters
    ----------
    dataset_key : str
        Dataset key configured in config.DATASETS.
    seed : int, default=42
        Random seed for reproducible splitting.
    force_download : bool, default=False
        If True, re-download dataset instead of using cached CSV.
    drop_sensitive_features : bool, default=False
        If True, exclude sensitive attributes from feature inputs.

    Returns
    -------
    PreparedData
        Prepared data container ready for model training and calibration.
    """
    from src.data_loader import load_dataset

    df = load_dataset(dataset_key, force_download=force_download)
    return preprocess_dataset(
        df=df,
        dataset_key=dataset_key,
        seed=seed,
        drop_sensitive_features=drop_sensitive_features,
    )
