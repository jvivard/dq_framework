import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from collections import Counter
import math
from sklearn.model_selection import train_test_split

def get_metrics(df_list,dataset_name_list):
    data=[]
    for df,dataset_name in zip(df_list,dataset_name_list):
        df=pd.DataFrame(df)
        #duplicate row percentage
        duplicate_row_percentage=((len(df)-len(df.drop_duplicates()))/len(df))*100
        for column in df.columns:
            #Meta info-:\n
            dataset_name=dataset_name
            column_name=column
            dtype=df[column].dtype
        
            #completness-:\n
            non_null_count=df[column].notnull().sum()
            missing_count=df[column].isnull().sum()
            missing_percent=(missing_count/len(df[column]))*100
        
            #Uniquness and variety\n
        
            unique_count=df[column].nunique()
            unique_percent=(unique_count/len(df[column]))*100
            mode_series = df[column].mode()
            mode = mode_series[0] if len(mode_series) > 0 else None
            mode_percent = ((df[column]==mode).sum()/len(df[column]))*100 if mode is not None else None
        
        
            #based on column types\n
        
            if is_numeric_dtype(df[column]):
                mean=df[column].mean()
                std=df[column].std()
                min=df[column].min()
                max=df[column].max()
                median=df[column].median()
                skewness=df[column].skew()
                abs_skew = abs(skewness)
                if abs_skew < 0.5:
                    skewness_level = "low"
                elif abs_skew < 1:
                    skewness_level = "moderate"
                else:
                    skewness_level = "high"
                col_vals = df[column].dropna()
                outliers = 0

                if len(col_vals) > 0:
                    abs_skew = abs(col_vals.skew())

                if abs_skew < 0.5:
                    # Z-score
                    z = (col_vals - col_vals.mean()) / col_vals.std()
                    outliers = int((np.abs(z) >= 3).sum())

                elif abs_skew < 1:
                    # IQR
                    Q1 = col_vals.quantile(0.25)
                    Q3 = col_vals.quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    outliers = int(((col_vals < lower) | (col_vals > upper)).sum())

                else:
                    # MAD (robust)
                    median = col_vals.median()
                    MAD = np.median(np.abs(col_vals - median))
                    if MAD > 0:
                        modified_z = 0.6745 * (col_vals - median) / MAD
                        outliers = int((np.abs(modified_z) > 3.5).sum())
                    else:
                        outliers = 0
            
                num_categories=None
                rare_category_percent=None
                whitespace_issues=None
                min_date=None
                max_date=None
                date_range_days=None
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                min_date = df[column].min()
                max_date = df[column].max()
                if pd.notna(min_date) and pd.notna(max_date):
                    date_range_days = (max_date - min_date).days
                else:
                    date_range_days = None

                mean=None; std=None; min=None; max=None; median=None
                outliers=None; num_categories=None; rare_category_percent=None
                whitespace_issues=None; skewness=None; skewness_level=None

            else:
                # categorical / text
                num_categories = unique_count
                vals = df[column].dropna().astype(str)
                freq = vals.value_counts(normalize=True) * 100
                rare = freq[freq <= 5]
                rare_category_percent = rare.sum() if not rare.empty else 0
                whitespace_issues = vals.str.startswith(" ").sum() + vals.str.endswith(" ").sum()

                mean=None; std=None; min=None; max=None; median=None; outliers=None
                min_date=None; max_date=None; date_range_days=None
                skewness=None; skewness_level=None

            data.append([dataset_name,column_name,dtype,non_null_count,missing_count,missing_percent,unique_count,unique_percent,mode,mode_percent,mean,std,skewness,skewness_level,min,max,median,outliers,num_categories,rare_category_percent,whitespace_issues,min_date,max_date,date_range_days,duplicate_row_percentage])
    return data        


def get_probs(df, column, bins):
    col_vals = df[column].dropna()

    if len(col_vals) == 0:
        return [1.0]

    probabilities = []

    if is_numeric_dtype(col_vals):
        # Clamp to reference bin range
        col_vals = col_vals.clip(bins[0], bins[-1])

        bin_indices = pd.cut(
            col_vals,
            bins=bins,
            include_lowest=True,
            labels=False
        )

        k = len(bins) - 1
        counts = Counter(bin_indices)

        for i in range(k):
            probabilities.append(counts.get(i, 0) / len(col_vals))

    else:
        bins = list(bins)
        counts = Counter(col_vals)
        for b in bins:
            probabilities.append(counts.get(b, 0) / len(col_vals))

    probs = np.array(probabilities)
    probs = probs + 1e-7
    probs = probs / probs.sum()
    return probs.tolist()

def KL_PSI_and_JS(df_base, df_new , column ):
        if is_numeric_dtype(df_base[column]):
            if len(df_base[column])<10000:
                bins = np.quantile(df_base[column], q=np.linspace(0,1,6))
                bins = np.unique(bins)
                bins = np.sort(bins)
                if len(bins) < 2:
                    return 0, 0, 0
            elif len(df_base[column])<50000:
                bins = np.quantile(df_base[column], q=np.linspace(0,1,11))
                bins = np.unique(bins)
                bins = np.sort(bins)
                if len(bins) < 2:
                    return 0, 0, 0
            else:
                bins = np.quantile(df_base[column], q=np.linspace(0,1,21))
                bins = np.unique(bins)
                bins = np.sort(bins)
                if len(bins) < 2:
                    return 0, 0, 0
        else:
            bins=df_base[column].unique()
        probs_base=get_probs(df_base , column, bins)
        probs_new=get_probs(df_new , column , bins)
        KL_div=0
        PSI_div=0
        JS_div=0
        probs_base=np.array(probs_base)
        probs_new=np.array(probs_new)
        M=(probs_new+probs_base)/2
        for i in range(0,len(probs_base) , 1):
            val_KL=probs_base[i]*(math.log(probs_base[i]/probs_new[i]))
            val_PSI=(probs_base[i]-probs_new[i])*(math.log(probs_base[i]/probs_new[i]))
            val_JS=(probs_base[i]*(math.log(probs_base[i]/M[i]))+probs_new[i]*(math.log(probs_new[i]/M[i])))/2
            KL_div+=val_KL
            PSI_div+=val_PSI
            JS_div+=val_JS
        return KL_div , PSI_div,JS_div
    

def get_entropy(df_base,column):
    if is_numeric_dtype(df_base[column]):
        if len(df_base[column])<10000:
            bins = np.quantile(df_base[column], q=np.linspace(0,1,6))
            bins = np.unique(bins)
            bins = np.sort(bins)
            if len(bins) < 2:
                return 0
        elif len(df_base[column])<50000:
            bins = np.quantile(df_base[column], q=np.linspace(0,1,11))
            bins = np.unique(bins)
            bins = np.sort(bins)
            if len(bins) < 2:
                return 0
        else:
            bins = np.quantile(df_base[column], q=np.linspace(0,1,21))
            bins = np.unique(bins)
            bins = np.sort(bins)
            if len(bins) < 2:
                return 0
    else:
        bins=df_base[column].unique()
    probs=get_probs(df_base,column,bins)
    entropy=0
    for prob in probs:
        entropy-=(prob*(math.log(prob)))
    if len(probs) <= 1:
        return 0
    return (entropy/math.log(len(probs)))

def get_type_error_count(df, column):
    error_count = 0
    col = df[column]

    if is_numeric_dtype(col):
        for v in col:
            try:
                float(v)
            except:
                error_count += 1

    elif pd.api.types.is_datetime64_any_dtype(col):
        for v in col:
            try:
                pd.to_datetime(v)
            except:
                error_count += 1

    else:  # categorical / object
        for v in col:
            if not isinstance(v, str):
                error_count += 1

    return error_count


def rule_based_errors(df,column,rules={}):
    errors=0
    rules=dict(rules)
    if column in rules.keys():
        for val in df[column]:
            if not(rules[column](val)):
                errors+=1
        return errors
    else:
        return 0

def advanced_metrics(dataset_name_list,df_base_list,df_new_list, rules={}):
    metrics=[]
    for df_base,df_new,dataset_name in zip(df_base_list,df_new_list,dataset_name_list):
        df_base=pd.DataFrame(df_base)
        df_new=pd.DataFrame(df_new)
        for column in df_base.columns:
            KL_div , PSI_div,JS_div =KL_PSI_and_JS(df_base , df_new,column)
            entropy=get_entropy(df_base,column)
            type_errors=get_type_error_count(df_base,column)
            rule_errors=rule_based_errors(df_base,column,rules)
        
            #more adv metrics soon
            metrics.append([dataset_name,column,KL_div,PSI_div,JS_div,entropy,type_errors,rule_errors])
    return metrics

def generate_meta(dataset_name_list,dataset_list,new_dataset_list=None, rules={}):
    data=get_metrics(dataset_list,dataset_name_list)
    df_old_list=[]
    df_new_list=[]
    if new_dataset_list is None:
        for df in dataset_list:
            df_old , df_new =train_test_split(df,test_size=0.3,random_state=1)
            df_old_list.append(df_old)
            df_new_list.append(df_new)
    else:
        if len(new_dataset_list) != len(dataset_list):
            raise ValueError("new_dataset_list must match dataset_list length")
        for i, (ref_df, new_df) in enumerate(zip(dataset_list,new_dataset_list)):

            if list(ref_df.columns) != list(new_df.columns):
                raise ValueError(
                f"Column mismatch in dataset index {i}"
            )

            if not all(ref_df.dtypes == new_df.dtypes):
                raise ValueError(
                f"Dtype mismatch in dataset index {i}"
            )
        df_old_list = [df for df in dataset_list]
        df_new_list = [df for df in new_dataset_list]
    metrics=advanced_metrics(dataset_name_list,df_old_list,df_new_list, rules=rules)
    meta_data=pd.DataFrame(data=data,columns=[
    # — Meta Info —\n
    "dataset_name",        
    "column_name",         
    "dtype",               

    # — Completeness —\n
    "non_null_count",      
    "missing_count",       
    "missing_percent",     

    # — Uniqueness & Variety —\n
    "unique_count",        
    "unique_percent",      
    "mode",                
    "mode_percent",        

    # — Numeric Columns —\n
    "mean",                
    "std",
    "skewness",
    "skewness_level",                 
    "min",                 
    "max",                 
    "median",              
    "outlier_count",       

    # — Categorical Columns —\n
    "num_categories",      
    "rare_category_percent",
    "whitespace_issues",   

    # — Datetime Columns —\n
    "min_date",            
    "max_date",            
    "date_range_days",     

    # — Quality & Resource —\n
    "duplicate_rows_percent"    
])
    meta_data.replace({pd.NaT: np.nan}, inplace=True)
    meta_table_adv=pd.DataFrame(data=metrics,columns=['dataset_name','column_name','KL_divergence','PSI_divergence','JS_divergence','Entropy','Type_Errors','rule_errors'])
    meta_table_combined=pd.merge(meta_data,meta_table_adv,on=['dataset_name', 'column_name'],how='inner')
    return meta_table_combined
