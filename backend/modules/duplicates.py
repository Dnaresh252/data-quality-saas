import pandas as pd


def analyze_duplicates(df):
    
    total_rows = len(df)
    duplicate_count = df.duplicated().sum()
    duplicate_percent = (duplicate_count / total_rows) * 100 if total_rows > 0 else 0
    
    duplicate_rows = df[df.duplicated(keep=False)]
    
    subset_duplicates = {}
    important_cols = [col for col in df.columns 
                     if 'id' not in col.lower() and 'index' not in col.lower()]
    
    if len(important_cols) > 0:
        subset_dup_count = df.duplicated(subset=important_cols, keep=False).sum()
        if subset_dup_count > duplicate_count:
            subset_duplicates = {
                "columns_checked": important_cols,
                "duplicate_count": int(subset_dup_count),
                "note": "More duplicates found when ignoring ID columns"
            }
    
    duplicate_groups = []
    if duplicate_count > 0 and duplicate_count < 1000:
        dup_df = df[df.duplicated(keep=False)].copy()
        dup_df['_dup_group'] = dup_df.groupby(list(df.columns)).ngroup()
        
        group_counts = dup_df['_dup_group'].value_counts().head(5)
        
        for group_id, count in group_counts.items():
            sample_indices = dup_df[dup_df['_dup_group'] == group_id].index.tolist()[:3]
            duplicate_groups.append({
                "group_size": int(count),
                "sample_indices": [int(idx) for idx in sample_indices]
            })
    
    def get_severity(pct):
        if pct == 0:
            return "None"
        elif pct < 1:
            return "Low"
        elif pct < 5:
            return "Medium"
        else:
            return "High"
    
    severity = get_severity(duplicate_percent)
    
    suggestions = []
    
    if duplicate_count == 0:
        suggestions.append("No duplicate rows detected. Dataset is clean.")
    elif duplicate_percent < 1:
        suggestions.append("Very few duplicates. Safe to remove using drop_duplicates()")
    elif duplicate_percent < 5:
        suggestions.append("Moderate duplicates found. Review if data represents transactions or events.")
    else:
        suggestions.append("High duplicate rate detected. Investigate data collection process.")
        suggestions.append("Check if duplicates are legitimate (e.g., repeated measurements) or errors.")
    
    if len(subset_duplicates) > 0:
        suggestions.append("Found more duplicates when ignoring ID columns. Consider subset-based deduplication.")
    
    return {
        "duplicate_count": int(duplicate_count),
        "duplicate_percent": round(duplicate_percent, 2),
        "total_rows": total_rows,
        "unique_rows": total_rows - duplicate_count,
        "severity": severity,
        "duplicate_groups": duplicate_groups,
        "subset_duplicates": subset_duplicates,
        "suggestions": suggestions
    }