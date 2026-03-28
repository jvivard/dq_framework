import pandas as pd
import numpy as np

def cleaning_recommendations(meta_data_combined):
    recs = []
    Thresholds = {
        "drop_missing": 20,
        "rare_category": 5,
        "small_date_range": 1
    }

    for i, row in meta_data_combined.iterrows():
        suggestions = []

        # 1. Drop early if hopeless
        if row['missing_percent'] > Thresholds['drop_missing']:
            suggestions.append(1)
        else:
            if pd.notna(row['whitespace_issues']) and row['whitespace_issues'] > 0:
                suggestions.append(4)

            if pd.notna(row['Type_Errors']) and row['Type_Errors'] > 0:
                suggestions.append(7)

            if pd.notna(row['rule_errors']) and row['rule_errors'] > 0:
                suggestions.append(8)

            if 0 < row['missing_percent'] <= Thresholds['drop_missing']:
                suggestions.append(2)

            if pd.notna(row['outlier_count']) and row['outlier_count'] > 0:
                suggestions.append(3)

            if pd.notna(row['rare_category_percent']) and row['rare_category_percent']>0:
                suggestions.append(5)

            if "datetime" in str(row['dtype']) and pd.notna(row['date_range_days']) and row['unique_count'] <= 1:
                suggestions.append(6)

            if row['skewness_level'] == 'high':
                suggestions.append(9)

            if not suggestions:
                suggestions.append(10)


        recs.append({
            "dataset_name": row["dataset_name"],
            "column_name": row["column_name"],
            "rule_ids": suggestions
        })

    return pd.DataFrame(recs)
