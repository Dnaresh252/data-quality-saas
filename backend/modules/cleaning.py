import pandas as pd
import numpy as np
import re


def handle_duplicates(df, duplicate_report, strategy="flag"):
    
    if not isinstance(duplicate_report, dict) or "duplicate_count" not in duplicate_report:
        df["_is_duplicate"] = df.duplicated()
        return df
    
    severity = duplicate_report.get("severity", "Low")
    
    if strategy == "keep":
        return df
    
    if strategy == "flag":
        df["_is_duplicate"] = df.duplicated()
        return df
    
    if strategy == "remove":
        return df.drop_duplicates().reset_index(drop=True)
    
    if strategy == "auto":
        if severity in ["High", "Medium"]:
            return df.drop_duplicates().reset_index(drop=True)
        else:
            df["_is_duplicate"] = df.duplicated()
            return df
    
    return df


def impute_missing_values(df, strategy="median"):
    
    cleaned = df.copy()
    
    for col in cleaned.columns:
        if cleaned[col].isna().sum() == 0:
            continue
        
        try:
            if cleaned[col].dtype in ['int64', 'float64', 'Int64', 'Float64']:
                if strategy == "median":
                    fill_value = cleaned[col].median()
                elif strategy == "mean":
                    fill_value = cleaned[col].mean()
                elif strategy == "zero":
                    fill_value = 0
                else:
                    fill_value = cleaned[col].median()
                
                cleaned[col] = cleaned[col].fillna(fill_value)
            
            elif str(cleaned[col].dtype).startswith('datetime'):
                cleaned[col] = cleaned[col].fillna(method='ffill').fillna(method='bfill')
            
            else:
                mode_values = cleaned[col].mode(dropna=True)
                if len(mode_values) > 0:
                    fill_value = mode_values[0]
                else:
                    fill_value = "Unknown"
                
                cleaned[col] = cleaned[col].fillna(fill_value)
        
        except Exception:
            continue
    
    return cleaned


def fix_data_types(df, dtype_report):
    
    if not isinstance(dtype_report, dict):
        return df
    
    cleaned = df.copy()
    
    date_cols = dtype_report.get("possible_date_columns", [])
    for col in date_cols:
        if col in cleaned.columns:
            try:
                cleaned[col] = pd.to_datetime(cleaned[col], errors='coerce')
            except:
                pass
    
    float_int_cols = dtype_report.get("float_int_candidates", [])
    for col in float_int_cols:
        if col in cleaned.columns:
            try:
                cleaned[col] = cleaned[col].astype('Int64')
            except:
                pass
    
    return cleaned


def normalize_text_columns(df, aggressive=False):
    
    cleaned = df.copy()
    
    try:
        text_cols = cleaned.select_dtypes(include=['object']).columns
    except:
        return cleaned
    
    for col in text_cols:
        try:
            cleaned[col] = cleaned[col].astype(str).str.strip()
            
            if aggressive:
                cleaned[col] = cleaned[col].str.lower()
                cleaned[col] = cleaned[col].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
            else:
                cleaned[col] = cleaned[col].apply(lambda x: re.sub(r'\s+', ' ', x))
        
        except:
            continue
    
    return cleaned


def handle_outliers(df, outlier_report, method="clip"):
    
    if not isinstance(outlier_report, dict):
        return df
    
    cleaned = df.copy()
    numeric_cols = outlier_report.get("numeric_columns", [])
    
    for col in numeric_cols:
        if col not in cleaned.columns:
            continue
        
        try:
            if method == "clip":
                Q1 = cleaned[col].quantile(0.25)
                Q3 = cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                cleaned[col] = cleaned[col].clip(lower=lower_bound, upper=upper_bound)
            
            elif method == "flag":
                Q1 = cleaned[col].quantile(0.25)
                Q3 = cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                cleaned[f"{col}_is_outlier"] = (
                    (cleaned[col] < lower_bound) | (cleaned[col] > upper_bound)
                )
            
            elif method == "remove":
                Q1 = cleaned[col].quantile(0.25)
                Q3 = cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                cleaned = cleaned[
                    (cleaned[col] >= lower_bound) & (cleaned[col] <= upper_bound)
                ]
        
        except:
            continue
    
    return cleaned.reset_index(drop=True)


def remove_constant_columns(df):
    
    cols_to_drop = []
    
    for col in df.columns:
        if df[col].nunique() == 1:
            cols_to_drop.append(col)
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    return df


def clean_dataset(
    df,
    dtype_report=None,
    inconsistency_report=None,
    outlier_report=None,
    duplicate_report=None,
    duplicate_strategy="flag",
    missing_strategy="median",
    outlier_method="clip",
    normalize_text=True,
    remove_constants=False
):
    
    cleaned_df = df.copy()
    
    cleaned_df = impute_missing_values(cleaned_df, strategy=missing_strategy)
    
    cleaned_df = handle_duplicates(
        cleaned_df,
        duplicate_report=duplicate_report,
        strategy=duplicate_strategy
    )
    
    if dtype_report:
        cleaned_df = fix_data_types(cleaned_df, dtype_report)
    
    if normalize_text:
        cleaned_df = normalize_text_columns(cleaned_df, aggressive=False)
    
    if outlier_report:
        cleaned_df = handle_outliers(cleaned_df, outlier_report, method=outlier_method)
    
    if remove_constants:
        cleaned_df = remove_constant_columns(cleaned_df)
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df