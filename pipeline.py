import pandas as pd
from dq_framework.generate_meta import generate_meta
from dq_framework.cleaning_recommendations import cleaning_recommendations
from dq_framework.get_cleaned_data import get_cleaned_data
from dq_framework.get_table_for_DQ_computation import get_table_for_DQ_computation
from dq_framework.summarize_dataset_health import summarize_dataset_health
from dq_framework.compute_dq_score import Compute_DQ_Score

def Run_DQ_Pipeline(dataset_names,dataset_list,new_dataset_list=None, rules={},interactive=False):
    meta_table_combined=generate_meta(dataset_names,dataset_list,new_dataset_list, rules=rules) #step 1   
    recommendations=cleaning_recommendations(meta_table_combined)#step 2
    dataset_list_cleaned,change_log=get_cleaned_data(dataset_names,dataset_list,recommendations, rules=rules,interactive=interactive) #step 3             
    meta_table_combined_cleaned=generate_meta(dataset_names,dataset_list_cleaned,new_dataset_list, rules=rules)# step 4 optional  
    
    # Note: Updated call to pass dataset_list explicitly as it is required by the extracted function
    main_metrics=get_table_for_DQ_computation(meta_table_combined, dataset_list, dataset_names)#step 5
    main_metrics_cleaned=get_table_for_DQ_computation(meta_table_combined_cleaned, dataset_list_cleaned, dataset_names)#step 6
    
    summarized_meta_table=summarize_dataset_health(meta_table_combined)#optional
    summarized_meta_table_cleaned=summarize_dataset_health(meta_table_combined_cleaned)#optional
    dirty_scores=Compute_DQ_Score(dataset_names,dataset_list,main_metrics)#step 7 
    cleaned_scores=Compute_DQ_Score(dataset_names,dataset_list_cleaned,main_metrics_cleaned)#step 8
    
    return { 
        "dirty_scores": pd.DataFrame(dirty_scores.items(),columns=["Dataset_name", "DQ_Score"]), 
        "cleaned_scores": pd.DataFrame(cleaned_scores.items(),columns=["Dataset_name", "DQ_Score"]),
        "cleaned_datasets":dataset_list_cleaned,
        "meta_table_before_cleaning": meta_table_combined, 
        "meta_table_after_cleaning":meta_table_combined_cleaned, 
        "recommendations": recommendations,
        "change_log": pd.DataFrame(change_log) if change_log else pd.DataFrame(), 
        "summarized_meta_before_cleaning": summarized_meta_table, 
        "summarized_meta_after_cleaning": summarized_meta_table_cleaned,
        "main_metrics_before_cleaning": main_metrics , 
        "main_metrics_after_cleaning": main_metrics_cleaned 
    }
