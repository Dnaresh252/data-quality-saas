import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp


def analyze_drift(df_old, df_new):
    
    numeric_cols_old = set(df_old.select_dtypes(include=['int64', 'float64', 'Int64', 'Float64']).columns)
    numeric_cols_new = set(df_new.select_dtypes(include=['int64', 'float64', 'Int64', 'Float64']).columns)
    numeric_cols = list(numeric_cols_old & numeric_cols_new)
    
    id_patterns = ['id', 'key', 'index', 'number', 'code']
    numeric_cols = [col for col in numeric_cols 
                   if not any(pattern in col.lower() for pattern in id_patterns)]
    
    if len(numeric_cols) == 0:
        return {
            "error": "No common numeric columns for drift analysis",
            "columns_checked": [],
            "drift_details": {},
            "suggestions": {}
        }
    
    drift_details = {}
    
    def classify_severity(val):
        if val < 0.1:
            return "None"
        elif val < 0.3:
            return "Low"
        elif val < 0.7:
            return "Medium"
        else:
            return "High"
    
    for col in numeric_cols:
        try:
            old_data = df_old[col].dropna()
            new_data = df_new[col].dropna()
            
            if len(old_data) < 10 or len(new_data) < 10:
                continue
            
            mean_old = old_data.mean()
            mean_new = new_data.mean()
            mean_drift = abs(mean_new - mean_old)
            mean_drift_pct = (mean_drift / abs(mean_old)) * 100 if mean_old != 0 else 0
            
            std_old = old_data.std()
            std_new = new_data.std()
            std_drift = abs(std_new - std_old)
            
            median_old = old_data.median()
            median_new = new_data.median()
            median_drift = abs(median_new - median_old)
            
            min_old = old_data.min()
            max_old = old_data.max()
            min_new = new_data.min()
            max_new = new_data.max()
            
            wasserstein_dist = wasserstein_distance(old_data, new_data)
            
            ks_stat, ks_pvalue = ks_2samp(old_data, new_data)
            
            normalized_drift = wasserstein_dist / (max_old - min_old) if (max_old - min_old) > 0 else 0
            
            drift_details[col] = {
                "mean_old": float(mean_old),
                "mean_new": float(mean_new),
                "mean_drift": float(mean_drift),
                "mean_drift_percent": round(float(mean_drift_pct), 2),
                
                "median_old": float(median_old),
                "median_new": float(median_new),
                "median_drift": float(median_drift),
                
                "std_old": float(std_old),
                "std_new": float(std_new),
                "std_drift": float(std_drift),
                
                "range_old": {"min": float(min_old), "max": float(max_old)},
                "range_new": {"min": float(min_new), "max": float(max_new)},
                
                "wasserstein_distance": float(wasserstein_dist),
                "normalized_drift": float(normalized_drift),
                
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "distribution_changed": ks_pvalue < 0.05,
                
                "severity": classify_severity(normalized_drift)
            }
        
        except Exception:
            continue
    
    categorical_cols_old = set(df_old.select_dtypes(include=['object', 'category']).columns)
    categorical_cols_new = set(df_new.select_dtypes(include=['object', 'category']).columns)
    categorical_cols = list(categorical_cols_old & categorical_cols_new)
    
    categorical_drift = {}
    
    for col in categorical_cols[:10]:
        try:
            old_dist = df_old[col].value_counts(normalize=True)
            new_dist = df_new[col].value_counts(normalize=True)
            
            all_categories = set(old_dist.index) | set(new_dist.index)
            
            old_dist = old_dist.reindex(all_categories, fill_value=0)
            new_dist = new_dist.reindex(all_categories, fill_value=0)
            
            psi = sum((new_dist - old_dist) * np.log((new_dist + 0.0001) / (old_dist + 0.0001)))
            
            new_categories = set(new_dist[new_dist > 0].index) - set(old_dist[old_dist > 0].index)
            missing_categories = set(old_dist[old_dist > 0].index) - set(new_dist[new_dist > 0].index)
            
            categorical_drift[col] = {
                "psi": float(psi),
                "severity": "High" if psi > 0.2 else "Medium" if psi > 0.1 else "Low",
                "new_categories": list(new_categories)[:5],
                "missing_categories": list(missing_categories)[:5],
                "category_count_old": len(old_dist[old_dist > 0]),
                "category_count_new": len(new_dist[new_dist > 0])
            }
        
        except Exception:
            continue
    
    suggestions = {}
    
    for col, info in drift_details.items():
        sev = info["severity"]
        
        if sev == "High":
            suggestions[col] = "Critical drift detected. Retrain models and investigate data pipeline changes."
        elif sev == "Medium":
            suggestions[col] = "Moderate drift detected. Monitor closely and validate model performance."
        elif sev == "Low":
            suggestions[col] = "Minor drift detected. Continue monitoring in production."
        else:
            suggestions[col] = "No significant drift detected."
        
        if info["distribution_changed"]:
            suggestions[col] += " Distribution shape has changed significantly (KS test)."
    
    for col, info in categorical_drift.items():
        if info["severity"] == "High":
            suggestions[col] = "High categorical drift. Review new/missing categories."
        elif len(info["new_categories"]) > 0:
            suggestions[col] = f"New categories detected: {', '.join(info['new_categories'])}"
    
    overall_severity = "Low"
    high_drift_cols = [col for col, info in drift_details.items() if info["severity"] == "High"]
    
    if len(high_drift_cols) > len(numeric_cols) * 0.3:
        overall_severity = "High"
    elif len(high_drift_cols) > 0:
        overall_severity = "Medium"
    
    return {
        "columns_checked": numeric_cols,
        "drift_details": drift_details,
        "categorical_drift": categorical_drift,
        "overall_severity": overall_severity,
        "high_drift_columns": high_drift_cols,
        "suggestions": suggestions
    }