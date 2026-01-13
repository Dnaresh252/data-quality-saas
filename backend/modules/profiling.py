import pandas as pd
import numpy as np


def profile_dataset(df):
    
    total_rows = len(df)
    total_columns = len(df.columns)
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'Int64', 'Float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    column_profiles = {}
    
    for col in df.columns:
        col_type = str(df[col].dtype)
        unique_count = df[col].nunique()
        null_count = df[col].isna().sum()
        null_pct = (null_count / total_rows) * 100
        
        profile = {
            "type": col_type,
            "unique_values": int(unique_count),
            "null_count": int(null_count),
            "null_percentage": round(float(null_pct), 2)
        }
        
        if col in numeric_cols:
            profile.update({
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
                "zeros": int((df[col] == 0).sum()),
                "negatives": int((df[col] < 0).sum())
            })
        
        if col in categorical_cols:
            value_counts = df[col].value_counts().head(10)
            top_values_dict = {str(k): int(v) for k, v in value_counts.items()}
            profile.update({
                "top_values": top_values_dict,
                "cardinality": "high" if unique_count > total_rows * 0.5 else "medium" if unique_count > 20 else "low"
            })
        
        if col in datetime_cols:
            profile.update({
                "min_date": str(df[col].min()),
                "max_date": str(df[col].max()),
                "date_range_days": int((df[col].max() - df[col].min()).days) if not df[col].isna().all() else 0
            })
        
        column_profiles[col] = profile
    
    duplicate_rows = df.duplicated().sum()
    
    potential_ids = []
    for col in df.columns:
        if df[col].nunique() == total_rows and not df[col].isna().any():
            potential_ids.append(col)
    
    constant_columns = []
    for col in df.columns:
        if df[col].nunique() == 1:
            constant_columns.append(col)
    
    high_cardinality_cats = []
    for col in categorical_cols:
        if df[col].nunique() > total_rows * 0.5:
            high_cardinality_cats.append(col)
    
    summary = {
        "shape": {
            "rows": total_rows,
            "columns": total_columns
        },
        "column_types": {
            "numeric": len(numeric_cols),
            "categorical": len(categorical_cols),
            "datetime": len(datetime_cols),
            "other": total_columns - len(numeric_cols) - len(categorical_cols) - len(datetime_cols)
        },
        "memory_usage_mb": round(memory_usage, 2),
        "duplicate_rows": int(duplicate_rows),
        "potential_id_columns": potential_ids,
        "constant_columns": constant_columns,
        "high_cardinality_categoricals": high_cardinality_cats
    }
    
    warnings = []
    
    if memory_usage > 100:
        warnings.append("Dataset is large (>100MB). Consider sampling or chunking for analysis.")
    
    if duplicate_rows > total_rows * 0.05:
        warnings.append(f"High number of duplicate rows ({duplicate_rows}). Review data source.")
    
    if len(constant_columns) > 0:
        warnings.append(f"Found {len(constant_columns)} constant columns. These can be removed.")
    
    if len(high_cardinality_cats) > 0:
        warnings.append(f"Found {len(high_cardinality_cats)} high-cardinality categorical columns. May need encoding strategy.")
    
    return {
        "summary": summary,
        "column_profiles": column_profiles,
        "warnings": warnings
    }