import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats


def analyze_outliers(df):
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'Int64', 'Float64']).columns.tolist()
    
    id_patterns = ['id', 'key', 'index', 'number', 'code']
    numeric_cols = [col for col in numeric_cols 
                   if not any(pattern in col.lower() for pattern in id_patterns)]
    
    if len(numeric_cols) == 0:
        return {
            "error": "No suitable numeric columns for outlier detection",
            "numeric_columns": [],
            "total_outliers": 0,
            "outliers_per_column": {},
            "suggestions": {}
        }
    
    df_numeric = df[numeric_cols].copy()
    
    for col in df_numeric.columns:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    
    df_numeric = df_numeric.dropna()
    
    if len(df_numeric) < 10:
        return {
            "error": "Insufficient data for outlier detection",
            "numeric_columns": numeric_cols,
            "total_outliers": 0,
            "outliers_per_column": {},
            "suggestions": {}
        }
    
    try:
        iso_forest = IsolationForest(
            contamination='auto',
            random_state=42,
            n_estimators=100
        )
        outlier_predictions = iso_forest.fit_predict(df_numeric)
        total_outliers = int((outlier_predictions == -1).sum())
    except Exception:
        total_outliers = 0
    
    outliers_per_column = {}
    outlier_indices = {}
    outlier_values = {}
    
    for col in numeric_cols:
        try:
            col_data = df[col].dropna()
            
            if len(col_data) < 4:
                outliers_per_column[col] = 0
                continue
            
            z_scores = np.abs(stats.zscore(col_data.astype(float)))
            z_outliers = (z_scores > 3).sum()
            
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
            
            outlier_count = int(max(z_outliers, iqr_outliers))
            outliers_per_column[col] = outlier_count
            
            outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
            outlier_idx = col_data[outlier_mask].index.tolist()
            outlier_vals = col_data[outlier_mask].tolist()
            
            outlier_indices[col] = outlier_idx[:10]
            outlier_values[col] = {
                "sample_values": [float(v) for v in outlier_vals[:5]],
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "Q1": float(Q1),
                "Q3": float(Q3)
            }
        
        except Exception:
            outliers_per_column[col] = 0
    
    suggestions = {}
    
    for col, count in outliers_per_column.items():
        if count == 0:
            suggestions[col] = "No outliers detected"
        elif count < 10:
            suggestions[col] = "Few outliers found. Review manually or apply IQR clipping"
        elif count < 50:
            suggestions[col] = "Moderate outliers detected. Consider IQR clipping or winsorization"
        else:
            outlier_pct = (count / len(df)) * 100
            if outlier_pct > 10:
                suggestions[col] = "High outlier rate. Investigate data source or distribution"
            else:
                suggestions[col] = "Many outliers detected. Use robust scaling or log transformation"
    
    severity = "Low"
    total_outlier_pct = (total_outliers / len(df)) * 100 if len(df) > 0 else 0
    
    if total_outlier_pct > 10:
        severity = "High"
    elif total_outlier_pct > 5:
        severity = "Medium"
    
    return {
        "numeric_columns": numeric_cols,
        "total_outliers": total_outliers,
        "outlier_percentage": round(total_outlier_pct, 2),
        "severity": severity,
        "outliers_per_column": outliers_per_column,
        "outlier_details": outlier_values,
        "suggestions": suggestions
    }