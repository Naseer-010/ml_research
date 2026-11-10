"""Data loading, standardization, and caching module for UCI benchmark datasets.

Downloads, standardizes, and caches the 4 UCI datasets used in the
Group-Wise Calibration in Compressed Tabular Models study:
- Adult (Census Income) [UCI ID: 2]
- Statlog German Credit [UCI ID: 144]
- Default of Credit Card Clients [UCI ID: 350]
- Bank Marketing [UCI ID: 222]

Supports fetching via ucimlrepo with automatic fallback to scikit-learn's
fetch_openml. Caches cleaned datasets to the data directory specified in config.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure project root is on sys.path for direct module execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(config.LOG_FORMAT)
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

# Mapping from UCI Statlog German Credit Attribute numbers to descriptive names
GERMAN_ATTRIBUTE_MAP: Dict[str, str] = {
    "Attribute1": "checking_status",
    "Attribute2": "duration",
    "Attribute3": "credit_history",
    "Attribute4": "purpose",
    "Attribute5": "credit_amount",
    "Attribute6": "savings_status",
    "Attribute7": "employment",
    "Attribute8": "installment_commitment",
    "Attribute9": "personal_status_and_sex",
    "Attribute10": "other_parties",
    "Attribute11": "residence_since",
    "Attribute12": "property_magnitude",
    "Attribute13": "age",
    "Attribute14": "other_payment_plans",
    "Attribute15": "housing",
    "Attribute16": "existing_credits",
    "Attribute17": "job",
    "Attribute18": "num_dependents",
    "Attribute19": "own_telephone",
    "Attribute20": "foreign_worker",
    "Attribute21": "class",
}

# OpenML fallback dataset IDs/names if ucimlrepo is unreachable
OPENML_DATASET_MAP: Dict[str, Any] = {
    "adult": 1590,  # OpenML Adult dataset ID
    "german_credit": 31,  # OpenML Statlog German Credit ID
    "credit_default": 42477,  # OpenML Default of Credit Card Clients ID
    "bank_marketing": 1461,  # OpenML Bank Marketing ID
}


def _fetch_from_ucirepo(uci_id: int) -> pd.DataFrame:
    """Fetch a dataset from the UCI Machine Learning Repository via ucimlrepo.

    Parameters
    ----------
    uci_id : int
        The integer repository identifier on UCI ML Repo.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame containing features and targets.

    Raises
    ------
    ImportError
        If ucimlrepo is not installed.
    Exception
        If the download or parsing fails.
    """
    from ucimlrepo import fetch_ucirepo

    logger.info("Fetching UCI dataset ID %d via ucimlrepo...", uci_id)
    repo = fetch_ucirepo(id=uci_id)

    X = repo.data.features
    y = repo.data.targets

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if y is None or (isinstance(y, pd.DataFrame) and y.empty):
        df = X.copy()
    elif isinstance(y, pd.DataFrame):
        df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    elif isinstance(y, pd.Series):
        df = X.copy().reset_index(drop=True)
        target_name = y.name if y.name is not None else "target"
        df[target_name] = y.reset_index(drop=True)
    else:
        df = X.copy().reset_index(drop=True)
        df["target"] = y

    # Deduplicate column names if any exist
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def _fetch_from_openml(dataset_key: str) -> pd.DataFrame:
    """Fallback fetcher using scikit-learn's fetch_openml.

    Parameters
    ----------
    dataset_key : str
        Dataset key configured in config.DATASETS.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame containing features and targets.

    Raises
    ------
    RuntimeError
        If openml fetch fails.
    """
    from sklearn.datasets import fetch_openml

    data_id = OPENML_DATASET_MAP.get(dataset_key)
    logger.info(
        "Attempting fallback fetch for '%s' from OpenML (data_id=%s)...",
        dataset_key,
        str(data_id),
    )

    try:
        if data_id is not None:
            dataset = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
        else:
            cfg = config.DATASETS[dataset_key]
            dataset = fetch_openml(name=cfg.name, as_frame=True, parser="auto")

        if hasattr(dataset, "frame") and dataset.frame is not None:
            df = dataset.frame.copy()
        else:
            X = dataset.data
            y = dataset.target
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            if isinstance(y, (pd.DataFrame, pd.Series)):
                df = pd.concat([X, y], axis=1)
            else:
                df = X.copy()
                df["target"] = y
        return df
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch dataset '{dataset_key}' from OpenML: {exc}"
        ) from exc


def _fetch_raw_dataset(dataset_key: str) -> pd.DataFrame:
    """Fetch raw dataset from ucimlrepo with automatic OpenML fallback.

    Parameters
    ----------
    dataset_key : str
        Dataset key configured in config.DATASETS.

    Returns
    -------
    pd.DataFrame
        Raw downloaded DataFrame.
    """
    if dataset_key not in config.DATASETS:
        raise ValueError(
            f"Unknown dataset key '{dataset_key}'. Expected one of {list(config.DATASETS.keys())}"
        )

    cfg = config.DATASETS[dataset_key]

    try:
        df = _fetch_from_ucirepo(cfg.uci_id)
        logger.info(
            "Successfully fetched '%s' via ucimlrepo (%d rows, %d cols).",
            dataset_key,
            len(df),
            len(df.columns),
        )
        return df
    except Exception as uci_exc:
        logger.warning(
            "ucimlrepo fetch failed for '%s' (UCI ID %d): %s. Trying OpenML fallback...",
            dataset_key,
            cfg.uci_id,
            uci_exc,
        )
        try:
            df = _fetch_from_openml(dataset_key)
            logger.info(
                "Successfully fetched '%s' via OpenML (%d rows, %d cols).",
                dataset_key,
                len(df),
                len(df.columns),
            )
            return df
        except Exception as openml_exc:
            raise RuntimeError(
                f"Failed to fetch dataset '{dataset_key}' from both ucimlrepo and OpenML. "
                f"ucimlrepo error: {uci_exc}; OpenML error: {openml_exc}"
            ) from openml_exc


def _strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespaces from column names and string cells.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with stripped strings and column names.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(
                lambda val: val.strip() if isinstance(val, str) else val
            )
    return df


def _clean_adult(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the Adult (Census Income) dataset.

    - Target column 'income' binarized: >50K -> 1, <=50K -> 0.
    - Sensitive attributes: 'sex' and 'race' are preserved and standardized.
    - Missing values represented as '?' are replaced with NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Adult DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned Adult DataFrame.
    """
    df = _strip_string_columns(df)
    cfg = config.DATASETS["adult"]

    # Match target column case-insensitively if needed
    target_candidates = ["income", "class", "salary", "income-per-year"]
    found_target: Optional[str] = None
    for cand in target_candidates:
        matches = [c for c in df.columns if c.lower() == cand]
        if matches:
            found_target = matches[0]
            break

    if found_target is None:
        raise KeyError(
            f"Could not locate target column for Adult dataset in columns: {list(df.columns)}"
        )

    if found_target != cfg.target_column:
        df = df.rename(columns={found_target: cfg.target_column})

    # Replace missing value placeholders
    df = df.replace(["?", "?.", "", "nan", "NaN"], np.nan)

    # Standardize target binarization (1 = >50K, 0 = <=50K)
    def binarize_income(val: Any) -> int:
        if pd.isna(val):
            return 0
        s = str(val).strip().rstrip(".")
        if s in [">50K", "1", "True", ">50k"]:
            return 1
        return 0

    df[cfg.target_column] = df[cfg.target_column].apply(binarize_income).astype(int)

    # Ensure sensitive columns ('sex', 'race') have lowercase column names
    col_rename = {}
    for col in df.columns:
        if col.lower() in ["sex", "race"]:
            col_rename[col] = col.lower()
    if col_rename:
        df = df.rename(columns=col_rename)

    for attr in cfg.sensitive_attributes:
        if attr not in df.columns:
            raise KeyError(
                f"Expected sensitive attribute '{attr}' not found in Adult dataset. "
                f"Available columns: {list(df.columns)}"
            )

    return df


def _clean_german_credit(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the Statlog German Credit dataset.

    - Maps Attribute1-Attribute20 column codes to descriptive feature names.
    - Derives 'sex' from Attribute9 (personal_status_and_sex).
    - Creates 'age_group' by binarizing age at 25 ('<25' vs '>=25').
    - Target 'class' binarized: 1 (good credit risk) -> 1, 2 (bad credit risk) -> 0.

    Parameters
    ----------
    df : pd.DataFrame
        Raw German Credit DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned German Credit DataFrame.
    """
    df = _strip_string_columns(df)
    cfg = config.DATASETS["german_credit"]

    # Map Attribute1..Attribute20 or A1..A20 if present
    rename_dict: Dict[str, str] = {}
    for col in df.columns:
        norm_col = col.capitalize()
        if norm_col in GERMAN_ATTRIBUTE_MAP:
            rename_dict[col] = GERMAN_ATTRIBUTE_MAP[norm_col]
        elif col.upper().startswith("A") and col[1:].isdigit():
            attr_key = f"Attribute{col[1:]}"
            if attr_key in GERMAN_ATTRIBUTE_MAP:
                rename_dict[col] = GERMAN_ATTRIBUTE_MAP[attr_key]

    if rename_dict:
        logger.info("Mapping German Credit column codes: %s", rename_dict)
        df = df.rename(columns=rename_dict)

    # Identify personal status and sex column
    personal_status_col: Optional[str] = None
    for cand in [
        "personal_status_and_sex",
        "personal_status",
        "Attribute9",
        "attribute9",
        "personal_status_sex",
    ]:
        if cand in df.columns:
            personal_status_col = cand
            break

    if personal_status_col is None:
        raise KeyError(
            f"Could not locate personal status column in German Credit. Columns: {list(df.columns)}"
        )

    # Derive 'sex' from personal status
    # A91: male : divorced/separated
    # A92: female : divorced/separated/married
    # A93: male : single
    # A94: male : married/widowed
    # A95: female : single
    def derive_sex(val: Any) -> str:
        s = str(val).strip().lower()
        if s.startswith("a92") or s.startswith("a95") or "female" in s:
            return "female"
        elif (
            s.startswith("a91")
            or s.startswith("a93")
            or s.startswith("a94")
            or "male" in s
        ):
            return "male"
        return "male"

    df["sex"] = df[personal_status_col].apply(derive_sex)

    # Identify age column and derive 'age_group' (<25 vs >=25)
    age_col: Optional[str] = None
    for cand in ["age", "Age", "Attribute13", "attribute13"]:
        if cand in df.columns:
            age_col = cand
            break

    if age_col is None:
        raise KeyError(
            f"Could not locate age column in German Credit. Columns: {list(df.columns)}"
        )

    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    df["age_group"] = np.where(df[age_col] >= 25, ">=25", "<25")

    # Locate and binarize target column 'class'
    target_col: Optional[str] = None
    for cand in ["class", "Class", "Attribute21", "credit_risk"]:
        if cand in df.columns:
            target_col = cand
            break

    if target_col is None:
        raise KeyError(
            f"Could not locate target column in German Credit. Columns: {list(df.columns)}"
        )

    if target_col != cfg.target_column:
        df = df.rename(columns={target_col: cfg.target_column})

    # In UCI German Credit: 1 = Good, 2 = Bad
    # Positive label in config is 1 (Good credit risk)
    def binarize_german_class(val: Any) -> int:
        s = str(val).strip().lower()
        if s in ["1", "1.0", "good"]:
            return 1
        elif s in ["2", "2.0", "bad", "0", "0.0"]:
            return 0
        return 1 if val == 1 else 0

    df[cfg.target_column] = (
        df[cfg.target_column].apply(binarize_german_class).astype(int)
    )

    for attr in cfg.sensitive_attributes:
        if attr not in df.columns:
            raise KeyError(
                f"Sensitive attribute '{attr}' missing from German Credit after cleaning."
            )

    return df


def _clean_credit_default(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the Default of Credit Card Clients dataset.

    - Sensitive attributes: 'SEX', 'EDUCATION', 'MARRIAGE' are preserved.
    - Target column 'default.payment' binarized: 1 (default) -> 1, 0 -> 0.
    - Dropping client ID identifier column if present.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Credit Default DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned Credit Default DataFrame.
    """
    df = _strip_string_columns(df)
    cfg = config.DATASETS["credit_default"]

    # Drop ID column if present
    id_cols = [c for c in df.columns if c.strip().upper() == "ID"]
    if id_cols:
        df = df.drop(columns=id_cols)

    # Find target column
    target_candidates = [
        "default.payment",
        "default.payment.next.month",
        "default payment next month",
        "default_payment_next_month",
        "y",
        "default",
    ]
    found_target: Optional[str] = None
    for cand in target_candidates:
        matches = [c for c in df.columns if c.lower() == cand]
        if matches:
            found_target = matches[0]
            break

    if found_target is None:
        # Check if last column is the target
        found_target = df.columns[-1]
        logger.warning(
            "Could not explicitly match Credit Default target; using last column: %s",
            found_target,
        )

    if found_target != cfg.target_column:
        df = df.rename(columns={found_target: cfg.target_column})

    # Standardize column casing for sensitive attributes (SEX, EDUCATION, MARRIAGE)
    col_rename = {}
    for col in df.columns:
        if col.upper() in ["SEX", "EDUCATION", "MARRIAGE"]:
            col_rename[col] = col.upper()
    if col_rename:
        df = df.rename(columns=col_rename)

    # Binarize target (0/1 integers)
    df[cfg.target_column] = (
        pd.to_numeric(df[cfg.target_column], errors="coerce")
        .fillna(0)
        .astype(int)
        .apply(lambda x: 1 if x == 1 else 0)
    )

    for attr in cfg.sensitive_attributes:
        if attr not in df.columns:
            raise KeyError(
                f"Sensitive attribute '{attr}' missing in Credit Default dataset. "
                f"Available columns: {list(df.columns)}"
            )

    return df


def _clean_bank_marketing(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the Bank Marketing dataset.

    - Creates 'age_group' by binarizing age at 30 ('<30' vs '>=30').
    - Sensitive attributes: 'age_group' and 'marital'.
    - Target column 'y' binarized: 'yes' -> 1, 'no' -> 0.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Bank Marketing DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned Bank Marketing DataFrame.
    """
    df = _strip_string_columns(df)
    cfg = config.DATASETS["bank_marketing"]

    # Identify target column 'y'
    target_col: Optional[str] = None
    for cand in ["y", "deposit", "subscribed"]:
        matches = [c for c in df.columns if c.lower() == cand]
        if matches:
            target_col = matches[0]
            break

    if target_col is None:
        target_col = df.columns[-1]

    if target_col != cfg.target_column:
        df = df.rename(columns={target_col: cfg.target_column})

    # Binarize target ('yes' -> 1, 'no' -> 0)
    def binarize_bank_target(val: Any) -> int:
        s = str(val).strip().lower()
        if s in ["yes", "y", "1", "1.0", "true"]:
            return 1
        return 0

    df[cfg.target_column] = (
        df[cfg.target_column].apply(binarize_bank_target).astype(int)
    )

    # Create age_group by binarizing age at 30 (<30 vs >=30)
    age_cols = [c for c in df.columns if c.lower() == "age"]
    if not age_cols:
        raise KeyError(
            f"Age column not found in Bank Marketing dataset. Columns: {list(df.columns)}"
        )
    age_col = age_cols[0]
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    df["age_group"] = np.where(df[age_col] >= 30, ">=30", "<30")

    # Standardize 'marital' column name
    marital_cols = [c for c in df.columns if c.lower() == "marital"]
    if marital_cols and marital_cols[0] != "marital":
        df = df.rename(columns={marital_cols[0]: "marital"})

    for attr in cfg.sensitive_attributes:
        if attr not in df.columns:
            raise KeyError(
                f"Sensitive attribute '{attr}' missing in Bank Marketing dataset."
            )

    return df


def _clean_dataset(df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """Route raw DataFrame to the dataset-specific cleaning routine.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset DataFrame.
    dataset_key : str
        Key identifying the dataset in config.DATASETS.

    Returns
    -------
    pd.DataFrame
        Cleaned and standardized DataFrame.
    """
    cleaners = {
        "adult": _clean_adult,
        "german_credit": _clean_german_credit,
        "credit_default": _clean_credit_default,
        "bank_marketing": _clean_bank_marketing,
    }

    cleaner = cleaners.get(dataset_key)
    if cleaner is None:
        raise ValueError(f"No cleaning routine defined for dataset '{dataset_key}'")

    return cleaner(df)


def load_dataset(dataset_key: str, force_download: bool = False) -> pd.DataFrame:
    """Load, standardize, and cache a dataset from config.DATASETS.

    If a cached CSV exists in config.DATA_DIR and force_download is False,
    the cached dataset is returned directly. Otherwise, the dataset is fetched,
    standardized (target binarized, sensitive columns derived), cached to CSV,
    and returned.

    Parameters
    ----------
    dataset_key : str
        Key of the dataset in config.DATASETS ('adult', 'german_credit',
        'credit_default', 'bank_marketing').
    force_download : bool, default=False
        If True, ignore existing cache and re-download/re-process.

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with binarized target and sensitive columns.

    Raises
    ------
    ValueError
        If dataset_key is not configured in config.DATASETS.
    RuntimeError
        If the dataset cannot be downloaded and no cache exists.
    """
    if dataset_key not in config.DATASETS:
        raise ValueError(
            f"Unknown dataset key: '{dataset_key}'. "
            f"Available keys: {list(config.DATASETS.keys())}"
        )

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = config.DATA_DIR / f"{dataset_key}.csv"

    # Load from cache if available
    if cache_path.exists() and not force_download:
        try:
            logger.info("Loading cached dataset '%s' from %s", dataset_key, cache_path)
            df_cached = pd.read_csv(cache_path)

            # Validate target column and sensitive attributes exist in cached file
            cfg = config.DATASETS[dataset_key]
            if cfg.target_column in df_cached.columns and all(
                attr in df_cached.columns for attr in cfg.sensitive_attributes
            ):
                return df_cached
            logger.warning(
                "Cached dataset '%s' is missing required columns. Re-processing...",
                dataset_key,
            )
        except Exception as exc:
            logger.warning(
                "Failed to read cache at %s: %s. Re-downloading...",
                cache_path,
                exc,
            )

    # Download raw dataset and clean it
    logger.info("Downloading and processing dataset '%s'...", dataset_key)
    df_raw = _fetch_raw_dataset(dataset_key)
    df_clean = _clean_dataset(df_raw, dataset_key)

    # Save cleaned dataset to cache
    try:
        df_clean.to_csv(cache_path, index=False)
        logger.info("Saved cleaned dataset '%s' to cache: %s", dataset_key, cache_path)
    except Exception as exc:
        logger.warning("Failed to save cache to %s: %s", cache_path, exc)

    return df_clean


def load_all_datasets(force_download: bool = False) -> Dict[str, pd.DataFrame]:
    """Load and standardize all 4 benchmark datasets.

    Parameters
    ----------
    force_download : bool, default=False
        If True, re-downloads all datasets even if cache exists.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping dataset keys to cleaned DataFrames.
    """
    datasets: Dict[str, pd.DataFrame] = {}
    for key in config.DATASETS:
        logger.info("=" * 60)
        logger.info("Loading dataset: %s", key)
        logger.info("=" * 60)
        datasets[key] = load_dataset(key, force_download=force_download)
    return datasets


if __name__ == "__main__":
    # Demonstration and quick sanity test when executed directly
    logger.info("Testing data_loader module...")
    all_data = load_all_datasets()
    for name, data in all_data.items():
        cfg = config.DATASETS[name]
        logger.info(
            "Dataset: %s | Shape: %s | Target '%s' values: %s",
            name,
            str(data.shape),
            cfg.target_column,
            str(data[cfg.target_column].value_counts().to_dict()),
        )
        for attr in cfg.sensitive_attributes:
            logger.info("  Sensitive '%s' groups: %s", attr, str(data[attr].unique()))
