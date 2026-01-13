import pandas as pd
import numpy as np


def analyze_missing_values(df):
    
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df)) * 100
    
    def get_severity(pct):
        if pct == 0:
            return "None"
        elif pct < 5:
            return "Low"
        elif pct < 20:
            return "Medium"
        else:
            return "High"
    
    def get_suggestion(col, pct, dtype):
        if pct == 0:
            return "No action needed"
        
        if pct > 50:
            return "Consider dropping this column due to high missing rate"
        
        if dtype in ['float64', 'int64', 'Int64', 'Float64']:
            return "Options: median imputation, mean imputation, or forward fill"
        elif str(dtype).startswith('datetime'):
            return "Options: forward fill, backward fill, or interpolation"
        else:
            return "Options: mode imputation, 'Unknown' category, or forward fill"
    
    details = {}
    
    for col in df.columns:
        count = int(missing_count[col])
        pct = float(missing_percent[col])
        
        if count > 0:
            details[col] = {
                "count": count,
                "percentage": round(pct, 2),
                "severity": get_severity(pct),
                "dtype": str(df[col].dtype),
                "suggestion": get_suggestion(col, pct, df[col].dtype)
            }
    
    total_missing_cells = missing_count.sum()
    total_cells = len(df) * len(df.columns)
    overall_missing_pct = (total_missing_cells / total_cells) * 100
    
    columns_with_missing = len([col for col in df.columns if missing_count[col] > 0])
    
    missing_patterns = []
    
    if columns_with_missing > 1:
        missing_mask = df.isna()
        pattern_counts = missing_mask.value_counts().head(5)
        
        for pattern, count in pattern_counts.items():
            if count > 1 and any(pattern):
                missing_cols = [df.columns[i] for i, val in enumerate(pattern) if val]
                if len(missing_cols) > 1:
                    missing_patterns.append({
                        "columns": missing_cols,
                        "occurrences": int(count),
                        "percentage": round((count / len(df)) * 100, 2)
                    })
    
    summary = {
        "total_missing_values": int(total_missing_cells),
        "overall_missing_percentage": round(overall_missing_pct, 2),
        "columns_with_missing": columns_with_missing,
        "total_columns": len(df.columns),
        "rows_with_any_missing": int(df.isna().any(axis=1).sum()),
        "complete_rows": int((~df.isna().any(axis=1)).sum())
    }
    
    recommendations = []
    
    high_missing_cols = [col for col, data in details.items() if data["percentage"] > 50]
    if high_missing_cols:
        recommendations.append(f"Consider dropping columns with >50% missing: {', '.join(high_missing_cols)}")
    
    if overall_missing_pct > 30:
        recommendations.append("Dataset has significant missing data. Investigate data collection process.")
    
    if len(missing_patterns) > 0:
        recommendations.append("Detected patterns in missing data. Missing values may not be random.")
    
    if columns_with_missing > len(df.columns) * 0.7:
        recommendations.append("Most columns have missing values. Review data quality at source.")
    
    return {
        "summary": summary,
        "details": details,
        "missing_patterns": missing_patterns,
        "recommendations": recommendations
    }