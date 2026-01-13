import pandas as pd
import chardet
import csv
from io import StringIO

from modules.missing_values import analyze_missing_values
from modules.duplicates import analyze_duplicates
from modules.data_types import analyze_data_types
from modules.inconsistencies import analyze_inconsistencies
from modules.outliers import analyze_outliers
from modules.drift import analyze_drift
from modules.cleaning import clean_dataset
from modules.profiling import profile_dataset
from modules.correlations import analyze_correlations


def detect_encoding(file_bytes):
    result = chardet.detect(file_bytes)
    encoding = result['encoding']
    confidence = result['confidence']
    
    if confidence < 0.7:
        encoding = 'utf-8'
    
    return encoding


def detect_delimiter(content_sample):
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(content_sample)
        return dialect.delimiter
    except:
        for delim in [',', ';', '\t', '|']:
            if delim in content_sample:
                return delim
        return ','


def load_csv_smart(filepath=None, file_bytes=None):
    if file_bytes is not None:
        encoding = detect_encoding(file_bytes[:10000])
        
        try:
            content = file_bytes.decode(encoding)
        except:
            content = file_bytes.decode('utf-8', errors='ignore')
        
        sample = content[:2048]
        delimiter = detect_delimiter(sample)
        
        df = pd.read_csv(
            StringIO(content),
            sep=delimiter,
            engine='python',
            on_bad_lines='skip'
        )
        
    elif filepath is not None:
        with open(filepath, 'rb') as f:
            raw_bytes = f.read()
        
        encoding = detect_encoding(raw_bytes[:10000])
        
        with open(filepath, 'r', encoding=encoding, errors='ignore') as f:
            sample = f.read(2048)
        
        delimiter = detect_delimiter(sample)
        
        df = pd.read_csv(
            filepath,
            sep=delimiter,
            encoding=encoding,
            engine='python',
            on_bad_lines='skip'
        )
    else:
        raise ValueError("Provide either filepath or file_bytes")
    
    return df


def run_full_pipeline(filepath=None, df_input=None, file_bytes=None, reference_df=None, duplicate_strategy="flag"):
    
    if df_input is not None:
        df = df_input.copy()
    elif file_bytes is not None:
        df = load_csv_smart(file_bytes=file_bytes)
    elif filepath is not None:
        df = load_csv_smart(filepath=filepath)
    else:
        raise ValueError("Provide filepath, df_input, or file_bytes")
    
    df_original = df.copy()
    
    profiling_report = profile_dataset(df)
    missing_report = analyze_missing_values(df)
    duplicate_report = analyze_duplicates(df)
    dtype_report = analyze_data_types(df)
    inconsistency_report = analyze_inconsistencies(df)
    outlier_report = analyze_outliers(df)
    correlation_report = analyze_correlations(df)
    
    if reference_df is not None:
        try:
            drift_report = analyze_drift(reference_df, df)
        except Exception as e:
            drift_report = {"error": f"Drift analysis failed: {str(e)}"}
    else:
        drift_report = None
    
    cleaned_df = clean_dataset(
        df=df,
        dtype_report=dtype_report,
        inconsistency_report=inconsistency_report,
        outlier_report=outlier_report,
        duplicate_report=duplicate_report,
        duplicate_strategy=duplicate_strategy
    )
    
    quality_score = calculate_quality_score(
        missing_report=missing_report,
        duplicate_report=duplicate_report,
        outlier_report=outlier_report,
        inconsistency_report=inconsistency_report
    )
    
    final_report = {
        "profiling": profiling_report,
        "missing_values": missing_report,
        "duplicates": duplicate_report,
        "data_types": dtype_report,
        "inconsistencies": inconsistency_report,
        "outliers": outlier_report,
        "correlations": correlation_report,
        "drift": drift_report,
        "quality_score": quality_score,
        "summary": {
            "rows_original": len(df_original),
            "rows_cleaned": len(cleaned_df),
            "columns": len(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
    }
    
    return final_report, cleaned_df


def calculate_quality_score(missing_report, duplicate_report, outlier_report, inconsistency_report):
    score = 100
    
    missing_penalty = 0
    for col_data in missing_report.get("details", {}).values():
        pct = col_data.get("percentage", 0)
        if pct > 20:
            missing_penalty += 10
        elif pct > 5:
            missing_penalty += 5
        elif pct > 0:
            missing_penalty += 2
    
    score -= min(missing_penalty, 30)
    
    dup_pct = duplicate_report.get("duplicate_percent", 0)
    if dup_pct > 5:
        score -= 15
    elif dup_pct > 1:
        score -= 8
    elif dup_pct > 0:
        score -= 3
    
    total_outliers = outlier_report.get("total_outliers", 0)
    if total_outliers > 100:
        score -= 10
    elif total_outliers > 50:
        score -= 5
    
    inconsistency_issues = (
        len(inconsistency_report.get("strip_issues", {})) +
        len(inconsistency_report.get("case_issues", {})) +
        len(inconsistency_report.get("special_char_issues", {}))
    )
    
    if inconsistency_issues > 5:
        score -= 15
    elif inconsistency_issues > 2:
        score -= 8
    elif inconsistency_issues > 0:
        score -= 4
    
    score = max(score, 0)
    
    if score >= 90:
        grade = "Excellent"
    elif score >= 75:
        grade = "Good"
    elif score >= 60:
        grade = "Fair"
    elif score >= 40:
        grade = "Poor"
    else:
        grade = "Critical"
    
    return {
        "score": score,
        "grade": grade,
        "breakdown": {
            "missing_data_impact": min(missing_penalty, 30),
            "duplicate_impact": 15 if dup_pct > 5 else (8 if dup_pct > 1 else (3 if dup_pct > 0 else 0)),
            "outlier_impact": 10 if total_outliers > 100 else (5 if total_outliers > 50 else 0),
            "inconsistency_impact": 15 if inconsistency_issues > 5 else (8 if inconsistency_issues > 2 else (4 if inconsistency_issues > 0 else 0))
        }
    }