#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#############################################
"""
    Author: xhuang
    Date created: 4/20/2019
    Date last modified: 4/25/2019
    Python Version: 3.7

    dependency: fastparquet, pandas , os
"""
#############################################


def read_parquet_partial(path_all, option=None, option_variable=None):
    """
    Example: 
        to read all parquets in path  
            ./Data_temp/tick/sample1/------code=001----xxx001.parquet
                                        |                      |--xxx002.parquet
                                        |                      |__ etc...
                                        |--code=002...
                                        |__etc...
        Input:            
            path_all =  ./Data_temp/tick/sample1/

            option: 'select': select some codes
                    'except': except these codes

            option_variable: only list of string, list of codes corresponding to the parametre option
        
        return: 
            pandas.DataFrame which contain one more column named  Ex: code, dtype is string
    """
    
    from fastparquet import ParquetFile
    import os
    import pandas as pd
    def read_multiparquet(path):
        df_tick_list = []
        for i in os.listdir(path):
            if i.lower().endswith('parquet'):
                df_tick_list.append(ParquetFile(os.path.join(path,i)).to_pandas())
        return pd.concat(df_tick_list, axis=0, ignore_index=True)
    
    val_list = [str(i.split('=')[1])  for i in os.listdir(path_all)]
    
    if option is None:
        option_variable = None
        dir_list_temp = [(i, str(i.split('=')[1]))  for i in os.listdir(path_all)]
    elif option == 'select':
        if type(option_variable) == list:
            # sub-category divided by '='
            dir_list_temp = [(i, str(i.split('=')[1]))  for i in os.listdir(path_all) if i.split('=')[1] in option_variable]
        else:
            raise TypeError('option_variable should be a list not <{}>'.format(type(option_variable)))
    elif option == 'except':
        if type(option_variable) == list:
            # sub-category divided by '='
            dir_list_temp = [(i, str(i.split('=')[1]))  for i in os.listdir(path_all) if i.split('=')[1] not in option_variable]
        else:
            raise TypeError('option_variable should be a list not <{}>'.format(type(option_variable)))
    
    dir_list = dir_list_temp
    col_name =  os.listdir(path_all)[0].split('=')[0]
    
    df_list = []
    for idx, (sub_dir, val) in enumerate(dir_list):
        df = read_multiparquet(os.path.join(path_all,sub_dir))
        df[col_name] = val
        df_list.append(df)
    return pd.concat(df_list, axis=0, ignore_index=True)
