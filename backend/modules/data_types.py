import pandas as pd
import numpy as np
import re


def analyze_data_types(df):
    
    raw_types = df.dtypes.astype(str).to_dict()
    
    mixed_type_columns = []
    possible_date_columns = []
    float_int_candidates = []
    possible_boolean_columns = []
    high_cardinality_strings = []
    
    for col in df.columns:
        unique_types = df[col].apply(lambda x: type(x).__name__).unique()
        if len(unique_types) > 1:
            mixed_type_columns.append({
                "column": col,
                "types_found": unique_types.tolist(),
                "current_dtype": str(df[col].dtype)
            })
    
    for col in df.columns:
        if df[col].dtype == 'object':
            non_null = df[col].dropna()
            
            if len(non_null) == 0:
                continue
            
            sample = non_null.head(100)
            
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',
                r'\d{2}/\d{2}/\d{4}',
                r'\d{2}-\d{2}-\d{4}',
                r'\d{4}/\d{2}/\d{2}'
            ]
            
            date_like_count = 0
            for pattern in date_patterns:
                date_like_count += sample.astype(str).str.contains(pattern, regex=True).sum()
            
            if date_like_count > len(sample) * 0.5:
                try:
                    pd.to_datetime(sample, errors='raise')
                    possible_date_columns.append(col)
                except:
                    pass
    
    for col in df.columns:
        if df[col].dtype == 'float64':
            non_null = df[col].dropna()
            
            if len(non_null) == 0:
                continue
            
            if (non_null % 1 == 0).all():
                float_int_candidates.append({
                    "column": col,
                    "reason": "All values are whole numbers",
                    "has_nulls": df[col].isna().any()
                })
    
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_vals = df[col].dropna().unique()
            
            if len(unique_vals) == 2:
                vals_lower = [str(v).lower() for v in unique_vals]
                
                boolean_patterns = [
                    {'yes', 'no'},
                    {'true', 'false'},
                    {'y', 'n'},
                    {'1', '0'},
                    {'t', 'f'}
                ]
                
                if set(vals_lower) in boolean_patterns:
                    possible_boolean_columns.append({
                        "column": col,
                        "values": unique_vals.tolist()
                    })
    
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_count = df[col].nunique()
            total_count = len(df)
            
            if unique_count > total_count * 0.8 and unique_count > 100:
                high_cardinality_strings.append({
                    "column": col,
                    "unique_count": int(unique_count),
                    "cardinality_ratio": round(unique_count / total_count, 2)
                })
    
    suggestions = {}
    
    for item in mixed_type_columns:
        col = item["column"]
        suggestions[col] = "Column has mixed data types. Clean or standardize values before conversion."
    
    for col in possible_date_columns:
        suggestions[col] = "Convert to datetime using pd.to_datetime() for time-based analysis."
    
    for item in float_int_candidates:
        col = item["column"]
        if item["has_nulls"]:
            suggestions[col] = "Convert to Int64 (nullable integer) to preserve null values."
        else:
            suggestions[col] = "Convert to int64 for memory efficiency."
    
    for item in possible_boolean_columns:
        col = item["column"]
        suggestions[col] = f"Convert to boolean. Values: {item['values']}"
    
    for item in high_cardinality_strings:
        col = item["column"]
        suggestions[col] = "High cardinality string. Consider if this should be a separate lookup table."
    
    type_summary = {
        "numeric": len(df.select_dtypes(include=['int64', 'float64', 'Int64', 'Float64']).columns),
        "categorical": len(df.select_dtypes(include=['object', 'category']).columns),
        "datetime": len(df.select_dtypes(include=['datetime64']).columns),
        "boolean": len(df.select_dtypes(include=['bool']).columns)
    }
    
    return {
        "raw_types": raw_types,
        "type_summary": type_summary,
        "mixed_type_columns": mixed_type_columns,
        "possible_date_columns": possible_date_columns,
        "float_int_candidates": [item["column"] for item in float_int_candidates],
        "float_int_details": float_int_candidates,
        "possible_boolean_columns": possible_boolean_columns,
        "high_cardinality_strings": high_cardinality_strings,
        "suggestions": suggestions
    }