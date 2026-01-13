import pandas as pd
import re


def analyze_inconsistencies(df):
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(categorical_cols) == 0:
        return {
            "summary": "No categorical columns found",
            "strip_issues": {},
            "case_issues": {},
            "special_char_issues": {},
            "mixed_alphanumeric": {},
            "rare_categories": {},
            "suggestions": {}
        }
    
    strip_issues = {}
    case_issues = {}
    special_char_issues = {}
    mixed_alnum_issues = {}
    rare_categories = {}
    inconsistent_formats = {}
    
    special_char_pattern = r'[^a-zA-Z0-9\s]'
    
    for col in categorical_cols:
        col_series = df[col].dropna().astype(str)
        
        if len(col_series) == 0:
            continue
        
        stripped = col_series.str.strip()
        strip_count = (col_series != stripped).sum()
        if strip_count > 0:
            strip_issues[col] = {
                "count": int(strip_count),
                "percentage": round((strip_count / len(col_series)) * 100, 2),
                "examples": col_series[col_series != stripped].head(3).tolist()
            }
        
        unique_original = col_series.nunique()
        unique_lower = col_series.str.lower().nunique()
        
        if unique_lower < unique_original:
            case_issues[col] = {
                "unique_original": int(unique_original),
                "unique_normalized": int(unique_lower),
                "potential_duplicates": int(unique_original - unique_lower)
            }
        
        has_special = col_series.str.contains(special_char_pattern, regex=True, na=False)
        special_count = has_special.sum()
        
        if special_count > 0:
            special_char_issues[col] = {
                "count": int(special_count),
                "percentage": round((special_count / len(col_series)) * 100, 2),
                "examples": col_series[has_special].head(3).tolist()
            }
        
        has_letters = col_series.str.contains(r'[A-Za-z]', regex=True, na=False)
        has_digits = col_series.str.contains(r'\d', regex=True, na=False)
        mixed_count = (has_letters & has_digits).sum()
        
        if mixed_count > 0 and mixed_count < len(col_series) * 0.9:
            mixed_alnum_issues[col] = {
                "count": int(mixed_count),
                "percentage": round((mixed_count / len(col_series)) * 100, 2)
            }
        
        value_counts = col_series.value_counts()
        rare = value_counts[value_counts < 5]
        
        if len(rare) > 0 and len(rare) < len(value_counts) * 0.5:
            rare_categories[col] = {
                "rare_count": len(rare),
                "total_unique": len(value_counts),
                "percentage_rare": round((len(rare) / len(value_counts)) * 100, 2),
                "examples": rare.head(5).to_dict()
            }
        
        lengths = col_series.str.len()
        length_std = lengths.std()
        
        if length_std > lengths.mean() * 0.5 and len(value_counts) > 10:
            inconsistent_formats[col] = {
                "min_length": int(lengths.min()),
                "max_length": int(lengths.max()),
                "mean_length": round(lengths.mean(), 2),
                "std_length": round(length_std, 2)
            }
    
    suggestions = {}
    
    for col in strip_issues:
        suggestions[col] = suggestions.get(col, [])
        suggestions[col].append("Remove leading/trailing whitespace with str.strip()")
    
    for col in case_issues:
        suggestions[col] = suggestions.get(col, [])
        suggestions[col].append("Normalize text case to lowercase or title case")
    
    for col in special_char_issues:
        suggestions[col] = suggestions.get(col, [])
        suggestions[col].append("Review special characters. Remove if not needed for business logic")
    
    for col in mixed_alnum_issues:
        suggestions[col] = suggestions.get(col, [])
        suggestions[col].append("Mixed alphanumeric values detected. Validate format consistency")
    
    for col in rare_categories:
        suggestions[col] = suggestions.get(col, [])
        suggestions[col].append("Group rare categories into 'Other' or investigate data quality")
    
    for col in inconsistent_formats:
        suggestions[col] = suggestions.get(col, [])
        suggestions[col].append("Inconsistent value lengths detected. Standardize format if possible")
    
    total_issues = (
        len(strip_issues) + 
        len(case_issues) + 
        len(special_char_issues) + 
        len(mixed_alnum_issues) + 
        len(rare_categories)
    )
    
    if total_issues == 0:
        severity = "None"
    elif total_issues <= 2:
        severity = "Low"
    elif total_issues <= 5:
        severity = "Medium"
    else:
        severity = "High"
    
    return {
        "summary": {
            "total_categorical_columns": len(categorical_cols),
            "columns_with_issues": total_issues,
            "severity": severity
        },
        "strip_issues": strip_issues,
        "case_issues": case_issues,
        "special_char_issues": special_char_issues,
        "mixed_alphanumeric": mixed_alnum_issues,
        "rare_categories": rare_categories,
        "inconsistent_formats": inconsistent_formats,
        "suggestions": suggestions
    }