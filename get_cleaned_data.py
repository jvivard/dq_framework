import pandas as pd
import numpy as np
from nltk import ngrams
from nltk.metrics.distance import jaccard_distance
from sklearn.preprocessing import PowerTransformer
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_categorical_dtype

def clean(df, column, clean_type,replacement=None, rules={}):
    def clean_outliers(df, column, replacement=None):
        col = df[column]

    # Guard: nothing to clean
        if col.dropna().std() == 0 or col.dropna().empty:
            return df[column]
        skewness_level = "low" if col.skew() < 0.5 else "moderate" if col.skew() < 1 else "high"
    # --- Choose method based on skewness ---
        if skewness_level == "low":
            # Z-score
            std = col.std()
            if std == 0 or np.isnan(std):
                return df[column]
            z = ((col - col.mean()) / std).abs()
            outlier_idx = z >= 3

        elif skewness_level == "moderate":
            # IQR
            Q1 = col.quantile(0.25)
            Q3 = col.quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0 or np.isnan(IQR):
                return df[column]
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlier_idx = (col < lower) | (col > upper)

        else:
            # MAD (robust)
            median = col.median()
            MAD = np.median(np.abs(col - median))
            if MAD == 0 or np.isnan(MAD):
                return df[column]
            modified_z = 0.6745 * (col - median) / MAD
            outlier_idx = np.abs(modified_z) > 3.5

    # --- Apply replacement ONCE ---
        if outlier_idx.sum() == 0:
            return df[column]

        if replacement is not None:
            df.loc[outlier_idx, column] = replacement
        else:
            df.loc[outlier_idx, column] = col.median()

        return df[column]

    def keep_string_else_none(val):
        if isinstance(val, str):
            return val
        else:
            return None
        #1 Drop column due to high missing values
    if clean_type == 1:
        df.drop(columns=[column], inplace=True)
        return None
        #2 Impute missing values
    elif clean_type == 2:
        if replacement:
            df[column] = df[column].fillna(replacement)
        else:
            if (is_numeric_dtype(df[column]) and (df[column].dtype != 'bool')) :
                df[column] = df[column].fillna(df[column].mean())
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                df[column]=df[column].fillna(df[column].mode()[0])
            else:
                df[column]=df[column].fillna(df[column].mode()[0])

        #3 Handle outliers using Z-score method
    elif clean_type == 3:
        df[column] = clean_outliers(df, column,replacement)  
        #4 Strip leading/trailing spaces
    elif clean_type == 4:
        def striping(value):
            if pd.isna(value):
                return value
            return str(value).strip()
        df[column] = df[column].apply(striping)

        #5 Combine rare categories and resolve typos
    elif clean_type == 5:
        if is_object_dtype(df[column]) or is_categorical_dtype(df[column]):
            df[column] = df[column].astype(str)

        counts = df[column].value_counts()
        total = len(df)

        substitutes = {}

        popular_categories = counts[(counts / total * 100) > 5].index.tolist()

        for k, v in counts.items():

            if pd.isna(k):
                continue

            val_pct = (v / total) * 100

            if val_pct <= 5:
                found_target = False
                k_str = k.lower().strip()

                if len(k_str) >= 3:
                    k_set = set("".join(g) for g in ngrams(k_str, 3))

                    for pop_val in popular_categories:
                        pop_str = pop_val.lower().strip()
                        pop_set = set("".join(g) for g in ngrams(pop_str, 3))

                        if k_set or pop_set:
                            dist = jaccard_distance(k_set, pop_set)
                            if dist <= 0.4:
                                substitutes[k] = pop_val
                                found_target = True
                                break

                if not found_target:
                    substitutes[k] = "Other"

        df[column] = df[column].replace(substitutes)
              
        #6 Check for incorrect/same date entries
    elif clean_type == 6:
        if (pd.api.types.is_datetime64_any_dtype(df[column]) and (df[column].nunique()<=1)):
            print( column , "has identical or invalid date entries not required for data analysis")
    #7 resolve missmatches and convert to nan
    elif clean_type == 7:
        col = df[column]
        if is_numeric_dtype(col):
            df[column] = pd.to_numeric(col, errors='coerce')
        elif pd.api.types.is_datetime64_any_dtype(col):
            df[column] = pd.to_datetime(col, errors='coerce')
        else:
            df[column] = col.apply(keep_string_else_none)
       
    #8 replace invalid buisness values (avoiding hallucinating of values)
    elif clean_type==8:
        func=rules[column]
        for i,val in enumerate(df[column]):
            if replacement:
                if not(func(val)):
                    df.loc[i,column]=replacement
            else:
                if not(func(val)):
                    df.loc[i,column]=None
    #9 Highly skewed distribution; consider power transformation for distance-based models
    elif clean_type == 9:
        col = df[column]
    # Safety checks
        if col.isna().all():
            return col
        if col.nunique() <= 1:
            return col
        if col.std() == 0:
            return col

        col_nonnull = col.dropna()
        if len(col_nonnull) < 3:
            return col

        # Re-check skewness after previous cleaning
        if abs(col_nonnull.skew()) < 1:
            return col

        pt = PowerTransformer(method="yeo-johnson")
        try:
            transformed = pt.fit_transform(col.values.reshape(-1,1)).ravel()
            return pd.Series(transformed, index=col.index)
        except Exception:
            return col

    elif clean_type == 10:
        pass
    return df[column]

def get_cleaned_data(dataset_name_list, dataset_list, recommendations, rules={}, interactive=False):
    NEEDS_REPLACEMENT = {2, 3, 8}  # 2 = impute, 3 = outliers, 8 = business rule

    dataset_list_cleaned = []
    change_log = []

    for item, dataset_name in zip(dataset_list, dataset_name_list):
        item = item.copy(deep=True)

        for column in item.columns:

            # fetch rule_ids instead of string suggestions
            rule_ids = recommendations[
                (recommendations['dataset_name'] == dataset_name) &
                (recommendations['column_name'] == column)
            ]['rule_ids'].iloc[0]

            for clean_type in rule_ids:

                replacement = None
                user_val = "default"

                if interactive:
                    print(f"\nDataset: {dataset_name}")
                    print(f"Column: {column}")
                    print(f"Rule ID: {clean_type}")
                    ans = input("Apply this fix? (yes/no): ").strip().lower()
                    if ans != "yes":
                        continue

                if clean_type in NEEDS_REPLACEMENT:
                    # col_dtype = item[column].dtype
                    # print(f"\nColumn: {column}")
                    # print(f"Suggested fix: {clean_type}")
                    # print(f"Column type: {col_dtype}")
                    # user_val = input("Enter replacement value (or type 'default'): ").strip()
                    user_val = "default"

                if user_val.lower() != "default":
                    try:
                        if pd.api.types.is_numeric_dtype(item[column]):
                            replacement = float(user_val)
                        else:
                            replacement = user_val
                    except:
                        print("Invalid value. Using default.")
                        replacement = None
                else:
                    replacement = None   # tells clean() to use default

                before = item[column].copy()

                result = clean(item, column, clean_type, replacement, rules=rules)

                if result is not None:
                    item[column] = result

                if result is not None and len(before) == len(result):
                    affected = (before != result).sum()
                else:
                    affected = 0

                change_log.append({
                    "dataset": dataset_name,
                    "column": column,
                    "rule_id": clean_type,
                    # rule_name intentionally omitted — can be added later via RULE_TEXT
                    "replacement": replacement if replacement is not None else "default",
                    "affected_rows": int(affected)
                })

        # dataset-level duplicate handling (kept as-is)
        if interactive:
            print("Do you want to remove any possible duplicates from dataset",dataset_name,"? (yes/no)")
            ans = input().strip().lower()
            if ans == "yes":
                before_len = len(item)
                item = item.drop_duplicates()
            after_len = len(item)

            dropped = before_len - after_len
            if dropped > 0:
                change_log.append({
        "dataset": dataset_name,
        "column": "__ROW_LEVEL__",   # indicates dataset-level operation
        "rule_id": 11,               # DROP_DUPLICATES
        "replacement": None,
        "affected_rows": int(dropped)
    })
        else:
            before_len = len(item)
            item = item.drop_duplicates()
            after_len = len(item)

            dropped = before_len - after_len
            if dropped > 0:
                change_log.append({
        "dataset": dataset_name,
        "column": "__ROW_LEVEL__",   # indicates dataset-level operation
        "rule_id": 11,               # DROP_DUPLICATES
        "replacement": None,
        "affected_rows": int(dropped)
    })

        dataset_list_cleaned.append(item)

    return dataset_list_cleaned, change_log

