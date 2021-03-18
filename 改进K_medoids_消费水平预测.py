# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 20:02:57 2021

@author: 92102
"""


import numpy as np
import pandas as pd

class cluster():
    def __init__(self):
        self.center=[]
        self.min_distent=[]
        self.cluster_member=[]
        
    def clear_center(self,):
        self.center=self.center.clear()
        
    def clear_min_distent(self,):
        self.center=self.center.clear()
        
    def clear_cluster_member(self,):
        self.center=self.center.clear()
        
    def set_center(self,user):
        self.center.append(user)
    
    def set_min_distent(self,user):
        self.center.append(user)
    
    def set_cluster_member(self,user):
        self.center.append(user)   
        
    


def variance_rank(data_sample):
    #求方差递增列表
    m,n=data_sample.shape
    distent_array=0
    
    #每个用户与其他用户的距离矩阵
    distent_array=data_sample
    for i in range(0,m):
        data_i=data_sample[i,:]
        distent=np.sqrt(np.sum(np.multiply(data_sample-data_i,data_sample-data_i), 1))
        distent=distent.T
        distent=distent
        if i==0:
            distent_array=distent            
        else :
            distent_array=np.vstack((distent_array,distent))
    
    #全局方差从小到大排序
    distent_std_array=[]
    for i in range(0,m):
        std_list=[]
        std_list.append(i)
        data_i=distent_array[i,:]
        distent=np.std(data_i)
        std_list.append(distent)
        distent_std_array.append(std_list)
    distent_std_array=sorted(distent_std_array,key=(lambda x:x[1]),reverse=False)
    
    return(distent_std_array)
    

def main():
    
    #数据读取
    consumption_evel_data = pd.read_excel("消费水平预测原始数据.xlsx")    
    data_sample = consumption_evel_data.iloc[:, 1:6].values
    
    #数据归一化
    mean = data_sample.mean(axis=0)
    std = data_sample.std(axis=0)
    data_sample = (data_sample-mean)/std

    

    
if __name__ == "__main__":
    main()