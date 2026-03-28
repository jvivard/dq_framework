import pandas as pd
import numpy as np
from scipy.stats import percentileofscore

def get_weights(dataset_name,summarized_meta):
    metrics=list(summarized_meta.columns)
    metrics.remove('skewness')
    metrics.remove('skewness_level')
    metrics.remove('dataset_name')
    metrics.remove('column_name')
    #while True:
        #print(f'enter your weights for each metric for {dataset_name} , they must sum to 1 , else metrics wont be considered\n')
    weights=[1/len(metrics)]*len(metrics)
        # for i,metric in enumerate(metrics):
        #     print("enter the weight for the following metric\n",metric+'\n','default value is ',1/len(metrics),'\n','press enter for default\n')
        #     weight=input()
        #     weight=weight.strip()
        #     if weight == "":
        #         continue
        #     elif weight.replace('.', '', 1).isdigit():
        #         weights[i] = float(weight)
        #     else:
        #         print("Invalid input, keeping default.\n")
        # weight_sum=sum(weights)
        # if abs(weight_sum - 1) < 1e-6:
        #     print("weights accepted")
        #     return weights
        #     break
        # else:
        #     print('weights not accepted , sum not 100 , please try again\n')
    return weights

def get_normalized_divergence(temp,metric):
    if metric=='PSI_divergence':
        vals=[]
        for index,row in temp.iterrows():
            value=row[metric]
            if(value<0.1):
                vals.append(1)
            elif(value>0.1 and value<0.25):
                vals.append(0.4)
            else:
                vals.append(0)
        temp[metric]=vals
        return temp[metric]
    else:
        penalties = {}
        for dataset, group in temp.groupby("dataset_name", sort=False):
            js_vals = group[metric]
            js_percentiles = [
                percentileofscore(js_vals, v, kind="weak") / 100
                for v in js_vals
            ]
            for idx, js_val, js_pct in zip(group.index, js_vals, js_percentiles):
                if js_val < 0.01:
                    penalties[idx] = 1
                else:
                    penalties[idx] = 1 - js_pct

        temp[metric] = temp.index.map(penalties)
        return temp[metric]


def get_new_values(summarized_meta_table):
    metrics=list(summarized_meta_table.columns)
    metrics.remove('dataset_name')
    metrics.remove('column_name')
    metrics.remove('skewness')
    metrics.remove('skewness_level')
    temp=summarized_meta_table.copy(deep=True)
    for i,metric in enumerate(metrics):
        if (metric=='PSI_divergence') or (metric=='JS_divergence'):
            temp[metric]=get_normalized_divergence(temp,metric)
        else:
            vals=[]
            for index,row in temp.iterrows():
                if((row[metric]<1)or(pd.isna(row[metric]))):
                    vals.append(1)
                elif(row[metric]<5):
                    vals.append(0.8)
                elif(row[metric]<10):
                    vals.append(0.4)
                else:
                    vals.append(0)
            temp[metric]=vals
    return temp
    
def Compute_DQ_Score(dataset_name_list,dataset_list,summarized_meta):
    metrics=list(summarized_meta.columns)
    metrics.remove('dataset_name')
    metrics.remove('column_name')
    metrics.remove('skewness')
    metrics.remove('skewness_level')
    temp=get_new_values(summarized_meta)
    scores={}
    for dataset_name,df in zip(dataset_name_list,dataset_list):
        weights=get_weights(dataset_name,summarized_meta)
        DQ_Score=[]
        for column in df.columns:
            value=0
            val=0
            for j,metric in enumerate(metrics):
                value=temp.loc[(temp['dataset_name']==dataset_name) & (temp['column_name']==column),metric].values[0]*weights[j]
                if pd.isna(value):
                    val += weights[j]
                else:
                    val+=min(value,weights[j])
            DQ_Score.append(val)
        DQ_Score=np.array(DQ_Score)
        DQ_Score=np.mean(DQ_Score)*100
        scores[dataset_name]=float(DQ_Score)
    return scores
