
def get_table_for_DQ_computation(meta_table, dataset_list, dataset_names): 
    # Added dataset_list and dataset_names arguments to resolve global dependency
    summarized_meta=meta_table.drop(columns=['dtype','non_null_count','missing_count','unique_count','mode','mode_percent','mean','std','min','max','median','num_categories','min_date','max_date','date_range_days','Entropy'])
    table_lengths={}
    for index,data in enumerate(dataset_list):
        table_lengths[dataset_names[index]]=len(data)
    outlier_percent=[]
    whitespace_percent=[]
    rule_errors_percent=[]
    Type_Errors_percent=[]
    for index,val in summarized_meta.iterrows():
        outlier_percent.append((val['outlier_count']/table_lengths[val['dataset_name']])*100)
        whitespace_percent.append((val['whitespace_issues']/table_lengths[val['dataset_name']])*100)
        rule_errors_percent.append((val['rule_errors']/table_lengths[val['dataset_name']])*100)
        Type_Errors_percent.append((val['Type_Errors']/table_lengths[val['dataset_name']])*100)
    summarized_meta['outlier_percent']=outlier_percent
    summarized_meta['whitespace_percent']=whitespace_percent
    summarized_meta['rule_errors_percent']=rule_errors_percent
    summarized_meta['Type_Errors_percent']=Type_Errors_percent
    summarized_meta.drop(columns=['outlier_count','whitespace_issues','rule_errors','Type_Errors','unique_percent','rare_category_percent','KL_divergence'],inplace=True)
    return summarized_meta
