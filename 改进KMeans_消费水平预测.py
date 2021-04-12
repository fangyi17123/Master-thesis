# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 07:53:02 2021

@author: 92102
"""
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

#必要的工具类
class NecessaryTools():  
    
    #比较两个一维数字元素列表是否一样
    def twolist(list1,list2):
        if len(list1)!=len(list2): 
            return False
        for nti3 in range (0,len(list1)):
            if(float(list1[nti3])!=float(list2[nti3])):
                return False
        return True
        
    #把聚类中心单独做一个列表，把聚类成员单独做一个列表,两个列表格式都是：[[],[]]
    def divide_cluster_list(nt_cluster_list,data_sample):
        nt_cluster_list=list(nt_cluster_list)
        nt_center_list=[]
        nt_member_list=[]
        nt_m,nt_n=data_sample.shape
    
        for nti4 in range(0,len(nt_cluster_list)):
            nt_center_list.append(nt_cluster_list[nti4][0])
    
        for nti5 in range(0,nt_m):
            nt_temp_list=data_sample[nti5,:].tolist()
            dcl_flag=True
            for ntj1 in range(0,len(nt_center_list)):
                if NecessaryTools.twolist(nt_temp_list,nt_center_list[ntj1]):
                    dcl_flag=False
                    break
            if (dcl_flag):
                nt_member_list.append(nt_temp_list)
    
        return(nt_center_list,nt_member_list)
        
    #把中心点列表转为矩阵,中心点格式是：[[],[]]
    def centerlist_centermat(cc_center_list):
        for cci1 in range(0,len(cc_center_list)):
            if cci1==0:
                cc_center_mat=np.mat(cc_center_list[cci1])
            else:
                cc_center_mat=np.row_stack((cc_center_mat,np.mat(cc_center_list[cci1])))
        return cc_center_mat
    
    #最近集族原则分配未分配集族成员元素
    def allocation_remain(ar_cluster_list,ar_center_list,ar_remain_member_list):      
        ar_center_mat=NecessaryTools.centerlist_centermat(ar_center_list)
        for ari in range(0,len(ar_remain_member_list)):
            ar_memberi_cluster_dif=ar_center_mat-np.mat(ar_remain_member_list[ari])
            ar_distence=np.sqrt(np.sum(np.multiply(ar_memberi_cluster_dif,ar_memberi_cluster_dif), 1))
            ar_distence=ar_distence.T
            ar_distence=ar_distence.tolist()
            ar_distence=ar_distence[0]
            ar_cluster_list[ar_distence.index(min(ar_distence))].append(ar_remain_member_list[ari])
            
        return(ar_cluster_list)

#获取初始K个聚类中心及相应聚类    
class GetInitialCluster():
    
    def __init__(self,k,data_sample):
        self.gic_cluster_list = []
        self.gic_k=k
        self.data_sample=data_sample
        self.gic_m,self.gic_n=self.data_sample.shape
              
    def find_minstd_poit(self):
        #每个用户与其他用户的距离矩阵
        poit=[]     
        distent_array=0
        for i in range(0,self.gic_m):
            data_i=self.data_sample[i,:]
            distent=np.sqrt(np.sum(np.multiply(self.data_sample-data_i,self.data_sample-data_i), 1))#按行相加
            distent=distent.T
            if i==0:
                distent_array=distent
            else :
                distent_array=np.row_stack((distent_array,distent))      
        poit.append((self.data_sample[np.argmin(np.std(distent_array, axis=1)),:]).tolist())         
        return(poit)
        
    def find_second_poit(self):
        poit_two=[]
        secon_poit_differ=self.data_sample-np.mat(self.gic_cluster_list[0][0])
        fsp_distent=np.sqrt(np.sum(np.multiply(secon_poit_differ,secon_poit_differ), 1))#按行相加
        fsp_distent=fsp_distent.T
        poit_two.append((self.data_sample[np.argmax(fsp_distent),:]).tolist())
        return(poit_two)
        

          
    def get_initial_cluster(self):
        self.gic_cluster_list.append(self.find_minstd_poit())
        self.gic_cluster_list.append(self.find_second_poit())

        
        #寻找剩余中心点,
        for gici1 in range(2,self.gic_k):
            gic_center_list,gic_member_list=NecessaryTools.divide_cluster_list(self.gic_cluster_list,self.data_sample)
            gic_center_list=list(gic_center_list)
            gic_member_list=list(gic_member_list)
            gic_center_mat=0
            gic_members_center_distence=0
            gic_members_center_distence_matrix=0
            
            #把中心点列表转为矩阵,中心点格式是：[[],[]]
            gic_center_mat=NecessaryTools.centerlist_centermat(gic_center_list)
            
            #每个成员点到已有中心的最小距离最大的那个点作为新中心成员点
            for frpi2 in range(0,len(gic_member_list)):
                gic_memberi_mat=np.mat(gic_member_list[frpi2]) 
                gic_members_center_distence=np.sqrt(np.sum(np.multiply(gic_center_mat-gic_memberi_mat,gic_center_mat-\
                                                                       gic_memberi_mat), 1))
                gic_members_center_mindistence_location=np.argmin(gic_members_center_distence)
                if frpi2==0:
                    gic_members_center_distence_matrix=gic_members_center_distence[gic_members_center_mindistence_location,:]
                else :
                    gic_members_center_distence_matrix=np.row_stack((gic_members_center_distence_matrix,\
                                                        gic_members_center_distence[gic_members_center_mindistence_location,:]))               
            gic_temp_list=[]  
            gic_temp_list.append(gic_member_list[np.argmax(gic_members_center_distence_matrix)])
            self.gic_cluster_list.append(gic_temp_list)      
        
        #把剩余的成员点分配到聚类中去
        gic_center_list=[]
        gic_member_list=[]
        gic_center_list,gic_member_list=NecessaryTools.divide_cluster_list(self.gic_cluster_list,self.data_sample)
        gic_center_list=list(gic_center_list)
        gic_member_list=list(gic_member_list)
        self.gic_cluster_list=NecessaryTools.allocation_remain(self.gic_cluster_list,gic_center_list,gic_member_list)
        
        return(self.gic_cluster_list)

#优化聚类
class GetBestCluster():
    
    def __init__(self,cluster_list,data_sample):
        self.gbc_cluster_list=list(cluster_list)
        self.data_sample=data_sample
        self.gbc_center_list,self.gbc_member_list=NecessaryTools.divide_cluster_list(self.gbc_cluster_list,self.data_sample)
        self.gbc_center_list_old=[]
        self.i=0#限制迭代次数
    
    def get_best_cluster(self):
               
        #两个聚类中心不变，即可判断为最优
        while (not(self.two_center_list(self.gbc_center_list,self.gbc_center_list_old))) and (self.i<50):

            self.i=self.i+1
            print('第',self.i,'轮迭代')
            self.gbc_center_list_old=list(self.gbc_center_list)
            gbc_cluster_list_temp=[]

            #到各成员点距离最小原则重新确定聚类中心
            for gbci2 in range(0,len(self.gbc_cluster_list)):
                gbc_members_all_distence_matrix=0
                gbc_clusteri_matrix=np.mat(self.gbc_cluster_list[gbci2])
                for gbcj2 in range(0,len(self.gbc_cluster_list[gbci2])):
                    gbc_membersi_matrix=np.mat(self.gbc_cluster_list[gbci2][gbcj2])
                    gbc_membersi_cluster_distence_dif=gbc_clusteri_matrix-gbc_membersi_matrix
                    gbc_membersi_cluster_distence_matrix=np.sqrt(np.sum(np.multiply(gbc_membersi_cluster_distence_dif,\
                                                                                    gbc_membersi_cluster_distence_dif), 1))
                    gbc_membersi_cluster_distence_matrix=np.sum(gbc_membersi_cluster_distence_matrix,0)
                    if gbcj2==0:
                        gbc_members_all_distence_matrix=gbc_membersi_cluster_distence_matrix
                    else:
                        gbc_members_all_distence_matrix=np.row_stack(( gbc_members_all_distence_matrix,\
                                                                      gbc_membersi_cluster_distence_matrix))
                gbc_cluster_list_temp2=[]
                gbc_cluster_list_temp2.append(self.gbc_cluster_list[gbci2][np.argmin(gbc_members_all_distence_matrix)])
                gbc_cluster_list_temp.append(gbc_cluster_list_temp2)
                    
            #把剩余的成员点分配到聚类中去
            self.gbc_center_list,gbc_member_list=NecessaryTools.divide_cluster_list(gbc_cluster_list_temp,self.data_sample)
            self.gbc_center_list=list(self.gbc_center_list)
            gbc_member_list=list(gbc_member_list) 
            self.gbc_cluster_list=NecessaryTools.allocation_remain(gbc_cluster_list_temp,self.gbc_center_list,gbc_member_list)
            
        return (self.gbc_cluster_list)

    
    #比较两个聚类列表是否一样,第一个元素是聚类中心,聚类中心格式是：[[中心]，[中心]，[中心]]
    def two_center_list(self,gbc_center_list1,gbc_center_list2):
        #聚类数量不同，即可判定不同
        if len(gbc_center_list1)!=len(gbc_center_list2):
            return False
        #如果一个聚类成员在另一个聚类成员里找不到，即可判断两个聚类不同
        for gbci1 in range(0,len(gbc_center_list1)):
            for gbcj1 in range(0,len(gbc_center_list1)):
                if NecessaryTools.twolist(gbc_center_list1[gbci1],gbc_center_list2[gbcj1]):
                    break;
                if gbcj1==(len(gbc_center_list1)-1):
                    return False;
        return(True)

def main():
    #程序入口，程序运行计时开始
    time_start=time.time()  
    k=4#分成四个聚类    
    cluster_list = []   #聚类列表
    
    #数据读取
    consumption_level_data = pd.read_excel("消费水平分析原始数据.xlsx")    
    data_sample = consumption_level_data.iloc[:, 0:2].values #去除用户序号
    data_sample_original = consumption_level_data.iloc[:,:].values

    
    print('你的数据是：')
    print(data_sample)
    plt.scatter(data_sample[:,0], data_sample[:,1])
    plt.show()
    
    m,n=data_sample.shape

    
    #获取初始聚类 
    gic = GetInitialCluster(k,data_sample)
    cluster_list = gic.get_initial_cluster()
    
    #聚类调优
    gbc = GetBestCluster(cluster_list,data_sample)
    cluster_list = gbc.get_best_cluster()
    
    #反归一化，还原，把输出聚类转变为元素初始格式
    consumption_level_list=[]
    consumption_level_list=data_sample_original.tolist()
    data_sample_list=[]
    data_sample_list = data_sample.tolist()
    cluster_list_temp=[]
    for i in range (0,len(cluster_list)):
        temp=[]
        for j in range (0,len(cluster_list[i])):
            temp.append(consumption_level_list[data_sample_list.index(cluster_list[i][j])])
        cluster_list_temp.append(temp)
    cluster_list=cluster_list_temp
    
    list1=[]
    list2=[]
    plt.scatter(cluster_list[0][0][0],cluster_list[0][0][1],s=200,c = 'r',marker='>')
    for i in range(1,len(cluster_list[0])):
        list1.append(cluster_list[0][i][0])
        list2.append(cluster_list[0][i][1])
    plt.scatter(list1,list2,c = 'b',marker='>')
    
    list1=[]
    list2=[]
    plt.scatter(cluster_list[1][0][0],cluster_list[1][0][1],s=200,c = 'r' ,marker='^')
    for i in range(1,len(cluster_list[1])):
        list1.append(cluster_list[1][i][0])
        list2.append(cluster_list[1][i][1])
    plt.scatter(list1,list2,c = 'g' ,marker='^')
    
    list1=[]
    list2=[]
    plt.scatter(cluster_list[2][0][0],cluster_list[2][0][1],s=200,c = 'r',marker='s')
    for i in range(1,len(cluster_list[2])):
        list1.append(cluster_list[2][i][0])
        list2.append(cluster_list[2][i][1])
    plt.scatter(list1,list2,c = 'k',marker='s')
    
    list1=[]
    list2=[]
    plt.scatter(cluster_list[3][0][0],cluster_list[3][0][1],s=200,c = 'r',marker='D')
    for i in range(1,len(cluster_list[3])):
        list1.append(cluster_list[3][i][0])
        list2.append(cluster_list[3][i][1])
    plt.scatter(list1,list2,c = 'y',marker='D')
    plt.show()

    
    print('聚类结果是(第一个元素是聚类中心，其它是聚类成员)：',cluster_list)
    
    #运行计时
    time_end=time.time()
    print('time cost',time_end-time_start,'s')

#程序主入口       
if __name__ == '__main__':
    main()
