"""
Dataset Management & Preprocessing Module.
Handles CSV upload, dataset profiling, column introspection, group identification, and robust preprocessing pipeline.
"""
import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from werkzeug.utils import secure_filename
from models.auth import get_db_connection
from config import Config

def binarize_sensitive_column(df, sensitive_col, privileged_group):
    """
    Binarizes sensitive attribute into 1 (Privileged) and 0 (Unprivileged).
    Supports:
    - Numeric Age columns (e.g. alter / Age):
      - Automatically splits into Older (Age >= threshold) vs Younger
    - General Categorical columns (Gender, Race, Religion, Nationality):
      - Privileged (1) matching privileged_group string (case-insensitive)
    - Fallback: Guarantees both 0 and 1 classes exist in output.
    """
    col_data = df[sensitive_col]
    
    # Detect if column is numeric or contains numbers
    is_numeric = pd.api.types.is_numeric_dtype(col_data)
    if not is_numeric and col_data.dtype == 'object':
        converted = pd.to_numeric(col_data, errors='coerce')
        if converted.notnull().sum() > 0.7 * len(col_data):
            col_data = converted
            is_numeric = True

    priv_str = str(privileged_group).strip()
    priv_str_lower = priv_str.lower()
    
    if is_numeric:
        thresh = 25.0
        numbers = re.findall(r'\d+\.?\d*', priv_str)
        if numbers:
            thresh = float(numbers[0])
        elif col_data.median() > 0:
            thresh = float(col_data.median())
            
        if 'young' in priv_str_lower or '<' in priv_str or 'minor' in priv_str_lower:
            res = (col_data < thresh).astype(int)
        else:
            res = (col_data >= thresh).astype(int)
    else:
        str_series = col_data.astype(str).str.strip().str.lower()
        
        if any(w in priv_str_lower for w in ['older', 'senior', 'adult', 'mature', '25']):
            res = (str_series.str.contains('older|senior|adult|mature|>|25', regex=True)).astype(int)
        else:
            res = (str_series == priv_str_lower).astype(int)
            if res.sum() == 0 or res.sum() == len(res):
                res = str_series.str.contains(re.escape(priv_str_lower), regex=True).astype(int)

    # Ultimate fallback: if result has only 1 unique class (e.g. all 0s or all 1s)
    if len(np.unique(res.values)) < 2:
        most_common = col_data.value_counts().index[0]
        res = (col_data == most_common).astype(int)
        if len(np.unique(res.values)) < 2:
            res = pd.Series(0, index=df.index)
            res.iloc[:len(res)//2] = 1

    return pd.Series(res, index=df.index)

def binarize_target_column(df, target_col):
    """
    Guarantees binary target vector y in {0, 1}.
    Handles binary categorical, numeric binary, and continuous/multiclass attributes.
    """
    target_series = df[target_col]
    unique_vals = target_series.dropna().unique()
    
    if len(unique_vals) <= 2:
        if pd.api.types.is_numeric_dtype(target_series):
            min_val = min(unique_vals)
            return (target_series > min_val).astype(int).values
        else:
            str_series = target_series.astype(str).str.strip().str.lower()
            pos_keywords = ['approved', 'yes', '1', 'good', 'grant', 'pass', 'high', 'true', '>50k', 'positive', 'accept', 'accepted']
            for kw in pos_keywords:
                if (str_series == kw).any():
                    return (str_series == kw).astype(int).values
                elif str_series.str.contains(re.escape(kw)).any():
                    return str_series.str.contains(re.escape(kw)).astype(int).values
            val1 = str(unique_vals[0]).strip().lower()
            return (str_series == val1).astype(int).values
    else:
        if pd.api.types.is_numeric_dtype(target_series):
            med = target_series.median()
            return (target_series >= med).astype(int).values
        else:
            str_series = target_series.astype(str).str.strip().str.lower()
            pos_keywords = ['approved', 'yes', '1', 'good', 'grant', 'pass', 'high', 'true', '>50k', 'positive', 'accept', 'accepted']
            for kw in pos_keywords:
                if (str_series == kw).any():
                    return (str_series == kw).astype(int).values
            top_val = str_series.value_counts().index[0]
            return (str_series == top_val).astype(int).values

import io

def save_uploaded_dataset(user_id, file_obj):
    """
    Saves uploaded CSV file, inspects shape, and creates dataset record in SQLite.
    Returns (success_flag, dataset_dict_or_error_msg)
    """
    if not file_obj or file_obj.filename == '':
        return False, "No file selected."
    
    if not file_obj.filename.lower().endswith('.csv'):
        return False, "Only CSV files are allowed."
    
    safe_name = secure_filename(file_obj.filename) or "uploaded_dataset.csv"
    filename = f"user_{user_id}_{safe_name}"
    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
    
    # Read CSV content string to save into database persistently
    csv_bytes = file_obj.read()
    csv_content = csv_bytes.decode('utf-8', errors='ignore')
    
    # Write to local/temp file
    with open(filepath, 'wb') as f:
        f.write(csv_bytes)
    
    try:
        df = pd.read_csv(io.StringIO(csv_content))
        row_count, col_count = df.shape
        if row_count < 1 or col_count < 2:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception:
                    pass
            return False, f"CSV file must contain at least 1 row and 2 columns (found {row_count} rows, {col_count} columns)."
        
        # Auto-detect target and sensitive columns defaults
        columns = list(df.columns)
        target_col = columns[-1] if columns else ''
        sensitive_col = columns[0] if columns else ''
        priv_group = 'Privileged'
        unpriv_group = 'Unprivileged'
        
        for col in columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['approved', 'target', 'class', 'label', 'outcome', 'kredit_risiko', 'credit_approved']):
                target_col = col
            if any(term in col_lower for term in ['alter', 'age', 'gender', 'sex', 'race', 'ethnicity']):
                sensitive_col = col

        # Infer group names based on sensitive column
        if any(term in sensitive_col.lower() for term in ['alter', 'age']):
            priv_group = 'Older (Age >= 25)'
            unpriv_group = 'Younger (Age < 25)'
        elif any(term in sensitive_col.lower() for term in ['gender', 'sex']):
            priv_group = 'Male'
            unpriv_group = 'Female'
        elif any(term in sensitive_col.lower() for term in ['race', 'ethnicity']):
            priv_group = 'White'
            unpriv_group = 'Non-White'
        else:
            unique_vals = list(df[sensitive_col].dropna().unique()) if sensitive_col in df else []
            if len(unique_vals) >= 2:
                priv_group = str(unique_vals[0])
                unpriv_group = str(unique_vals[1])

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO datasets 
               (user_id, filename, filepath, csv_content, row_count, column_count, target_column, sensitive_column, privileged_group, unprivileged_group)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, filename, filepath, csv_content, row_count, col_count, target_col, sensitive_col, priv_group, unpriv_group)
        )
        conn.commit()
        dataset_id = cursor.lastrowid
        conn.close()
        
        return True, get_dataset_by_id(dataset_id)
    except Exception as e:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass
        return False, f"Failed to process CSV: {str(e)}"

def get_dataset_by_id(dataset_id):
    """Retrieves dataset record from database by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    ds = cursor.fetchone()
    conn.close()
    return dict(ds) if ds else None

def get_user_datasets(user_id):
    """Retrieves all datasets uploaded by a specific user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets WHERE user_id = ? ORDER BY uploaded_at DESC", (user_id,))
    datasets = cursor.fetchall()
    conn.close()
    return [dict(d) for d in datasets]

def load_dataset_dataframe(filepath_or_ds):
    """Loads CSV file into Pandas DataFrame with fallback path searches and database csv_content."""
    filepath = filepath_or_ds if isinstance(filepath_or_ds, str) else (filepath_or_ds.get('filepath') if isinstance(filepath_or_ds, dict) else "")
    
    if filepath and os.path.exists(filepath):
        return pd.read_csv(filepath)
    
    # Fallbacks for serverless environments / relative paths
    filename = os.path.basename(filepath) if filepath else ""
    possible_paths = [
        os.path.join(Config.UPLOAD_FOLDER, filename),
        os.path.join(Config.BASE_DIR, 'static', 'uploads', filename),
        os.path.join(Config.BASE_DIR, filename)
    ]
    for p in possible_paths:
        if p and os.path.exists(p):
            return pd.read_csv(p)
            
    # Try retrieving from database csv_content if serverless container reset wiped physical file
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT csv_content, filepath FROM datasets WHERE filepath = ? OR filename = ?", (filepath, filename))
    row = cursor.fetchone()
    conn.close()
    
    if row and row['csv_content']:
        content = row['csv_content']
        # Optionally rewrite back to temp path
        try:
            target_p = row['filepath'] or os.path.join(Config.UPLOAD_FOLDER, filename)
            os.makedirs(os.path.dirname(target_p), exist_ok=True)
            with open(target_p, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception:
            pass
        return pd.read_csv(io.StringIO(content))
            
    raise FileNotFoundError(f"Dataset file not found at path: {filepath}")



def inspect_dataset(filepath):
    """
    Returns structured analysis of dataset: column names, data types, missing values, sample head rows.
    """
    df = load_dataset_dataframe(filepath)
    preview_rows = df.head(10).to_dict(orient='records')
    columns = list(df.columns)
    column_stats = {}
    
    for col in columns:
        unique_vals = list(df[col].dropna().unique())[:30]
        unique_vals_str = [str(v) for v in unique_vals]
        column_stats[col] = {
            'dtype': str(df[col].dtype),
            'missing': int(df[col].isnull().sum()),
            'unique_count': int(df[col].nunique()),
            'sample_values': unique_vals_str
        }
        
    return {
        'total_rows': len(df),
        'total_cols': len(df.columns),
        'columns': columns,
        'column_stats': column_stats,
        'preview': preview_rows
    }

def update_dataset_config(dataset_id, target_col, sensitive_col, privileged_group, unprivileged_group):
    """Updates target, sensitive attribute, and privileged/unprivileged group settings for dataset."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """UPDATE datasets 
           SET target_column = ?, sensitive_column = ?, privileged_group = ?, unprivileged_group = ?
           WHERE id = ?""",
        (target_col, sensitive_col, str(privileged_group), str(unprivileged_group), dataset_id)
    )
    conn.commit()
    conn.close()
    return True

def preprocess_dataset(filepath, target_col, sensitive_col, privileged_group):
    """
    Preprocesses dataset for model training & fairness analysis.
    - Encodes target into binary (0/1)
    - Encodes sensitive attribute into binary privileged (1) vs unprivileged (0)
    - Encodes categorical features
    - Imputes missing numerical values
    - Standardizes features
    Returns dict containing processed data splits & encoders.
    """
    df = load_dataset_dataframe(filepath)
    cols = list(df.columns)
    
    if len(cols) < 2:
        raise ValueError(f"Dataset must contain at least 2 columns (found {len(cols)}).")

    if target_col and sensitive_col and str(target_col).strip() == str(sensitive_col).strip():
        raise ValueError(f"Target column and Sensitive Attribute column cannot be the same ('{target_col}'). Please select different columns in Dataset Configuration.")

    # Robust fallback for target_col and sensitive_col if None or missing
    if not target_col or target_col not in cols:
        target_col = cols[-1] if cols else ''
    if not sensitive_col or sensitive_col not in cols:
        sensitive_col = cols[0] if (cols and cols[0] != target_col) else (cols[1] if len(cols) > 1 else cols[0])
    if not privileged_group:
        privileged_group = 'Male' if 'gender' in str(sensitive_col).lower() else 'Older'

    df = df.dropna(subset=[target_col, sensitive_col]).copy()
    
    # Process Sensitive Attribute
    sensitive_binary = binarize_sensitive_column(df, sensitive_col, privileged_group).values
    
    # Process Target Attribute
    y = binarize_target_column(df, target_col)
        
    # Features dataframe (drop target)
    X_df = df.drop(columns=[target_col]).copy()
    
    # Identify numerical and categorical columns
    cat_cols = X_df.select_dtypes(include=['object', 'category']).columns
    num_cols = X_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    
    # Impute & encode categorical features using pd.get_dummies
    X_processed = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)
    
    # Impute numerical features if any missing
    if num_cols.any():
        imputer = SimpleImputer(strategy='median')
        X_processed[num_cols] = imputer.fit_transform(X_processed[num_cols])
        
    # Safe Train / Test split stratification check
    strat = y if (len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2) else None
    
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X_processed, y, sensitive_binary, test_size=0.25, random_state=42, stratify=strat
    )
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'A_train': A_train,
        'A_test': A_test,
        'feature_names': list(X_processed.columns),
        'X_train_df': X_train,
        'X_test_df': X_test
    }

