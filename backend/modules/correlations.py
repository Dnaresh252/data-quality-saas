import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


def analyze_correlations(df):
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'Int64', 'Float64']).columns.tolist()
    
    id_like_patterns = ['id', 'key', 'index', 'number']
    numeric_cols = [col for col in numeric_cols 
                   if not any(pattern in col.lower() for pattern in id_like_patterns)]
    
    if len(numeric_cols) < 2:
        return {
            "numeric_correlations": {},
            "high_correlations": [],
            "categorical_associations": {},
            "warnings": ["Not enough numeric columns for correlation analysis"]
        }
    
    corr_matrix = df[numeric_cols].corr()
    
    correlations_dict = {}
    for col in corr_matrix.columns:
        correlations_dict[col] = corr_matrix[col].to_dict()
    
    high_correlations = []
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            corr_value = corr_matrix.loc[col1, col2]
            if abs(corr_value) > 0.7 and not pd.isna(corr_value):
                high_correlations.append({
                    "column1": col1,
                    "column2": col2,
                    "correlation": round(float(corr_value), 3),
                    "strength": "strong positive" if corr_value > 0.7 else "strong negative"
                })
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if df[col].nunique() < 50 and df[col].nunique() > 1]
    
    categorical_associations = {}
    
    if len(categorical_cols) >= 2:
        for i, col1 in enumerate(categorical_cols[:5]):
            for col2 in categorical_cols[i+1:6]:
                try:
                    contingency_table = pd.crosstab(df[col1], df[col2])
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    
                    if p_value < 0.05:
                        n = contingency_table.sum().sum()
                        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                        
                        categorical_associations[f"{col1}_vs_{col2}"] = {
                            "chi2_statistic": round(float(chi2), 4),
                            "p_value": round(float(p_value), 4),
                            "cramers_v": round(float(cramers_v), 3),
                            "significant": True,
                            "association_strength": "strong" if cramers_v > 0.5 else "moderate" if cramers_v > 0.3 else "weak"
                        }
                except:
                    continue
    
    warnings = []
    
    if len(high_correlations) > 0:
        warnings.append(f"Found {len(high_correlations)} pairs of highly correlated features. Consider feature selection.")
    
    multicollinear_features = []
    for corr_pair in high_correlations:
        if abs(corr_pair["correlation"]) > 0.9:
            multicollinear_features.append(f"{corr_pair['column1']} and {corr_pair['column2']}")
    
    if len(multicollinear_features) > 0:
        warnings.append(f"Potential multicollinearity detected. Review: {', '.join(multicollinear_features)}")
    
    return {
        "numeric_correlations": correlations_dict,
        "high_correlations": high_correlations,
        "categorical_associations": categorical_associations,
        "warnings": warnings
    }