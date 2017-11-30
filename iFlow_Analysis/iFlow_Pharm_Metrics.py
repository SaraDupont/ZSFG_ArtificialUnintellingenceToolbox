#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 09:58:19 2017

@author: davidbmccoy
"""
from os import listdir
from os.path import isfile, join
import pandas as pd
import re 
import statsmodels.api as sm
import numpy as np 
from sklearn import metrics
import operator
from scipy.stats import kurtosis, skew, kstest 
import argparse


def get_parser():
    
    parser = argparse.ArgumentParser(description="Calculate pharmacokinetic metrics from iFlow csv data and produce summary table")
    parser.add_argument("-data",
                        help="Path to csv data folder",
                        type=str,
                        dest="path")
    
    parser.add_argument("-roi",
                        help="Number of ROIs used to get data from iFlow",
                        type=int,
                        dest="ROI",
                        default=1)
    
    parser.add_argument("-out_file",
                    help="Output filename",
                    type=str,
                    dest="out_file",
                    default="iFlow_Analysis")
    return parser

    
class iFlow_Analysis():
    
    def __init__(self, param):
        
            self.param = param # parser parameters
            self.list_subj = [] 
            
            self.patient_name_DF = []
            self.patient_mrn_DF = [] 
            self.patient_birthdate_DF = [] 
            self.patient_sex_DF = []
            self.procedure_date_DF = [] 
            self.procedure_time_DF = [] 
            
            self.ROI_Area = [] 
            self.ROI_Ratio_Peak = [] 
            self.ROI_Peak_Time = [] 
            self.ROI_Ratio_AUC = [] 
            self.ROI_Number = [] 
            self.ROI_Names = [] 
            
            self.halflife = []
            self.double_time = [] 
            self.auc = [] 
            self.elimination_rate = [] 
            self.TOMC = [] 
            self.CMAX = [] 
            self.kurtosis = []
            self.skewness = []
            self.ktest = []
            
            
    def hasNumbers(self,inputString):
        return any(char.isdigit() for char in inputString)

    def processing(self): 
        self.list_subj = [f for f in listdir(self.param.path) if isfile(join(self.param.path, f)) and f.endswith(".csv")]
        for file in self.list_subj: 
            
            data = join(self.param.path, file)
            #df = pd.read_csv(open(data,'rU'), encoding='utf-8', engine='c')
            df = pd.read_csv(data, engine="python", header=None, delimiter=";|,")
            m = re.search('START', str(df.index.values[0]))
            
            if m: 
                df.index.name = 'new_index'
                df.reset_index(inplace=True)
                
                
            index_Patient = [i for i, s in enumerate(df.iloc[:,0]) if 'Patient' in str(s)]
            index_Acquisition = [i for i, s in enumerate(df.iloc[:,0]) if 'Acquisition' in str(s)]
            index_ROI = [i for i, s in enumerate(df.iloc[:,0]) if 'ROI' in str(s)]
            index_Table = [i for i, s in enumerate(df.iloc[:,0]) if 'TABLE' in str(s)]
            
            
            self.patient_name = df.iloc[index_Patient,1].item()
            print("Processing patient"+self.patient_name)

            self.patient_mrn = df.iloc[index_Patient,2].item()
            self.patient_birthdate = df.iloc[index_Patient,3].item()
            self.patient_sex = df.iloc[index_Patient,4].item()
            
            self.procedure_date = df.iloc[index_Acquisition,1].item()
            self.procedure_time = df.iloc[index_Acquisition,2].item()
            
            roi_summaries = df.iloc[index_ROI[0]+1:index_ROI[0]+self.param.ROI+1, 1:5]
            roi_summaries.columns = ['ROI Area', 'ROI Peak/Ref Peak','ROI Peak Time', 'ROI AUC/Ref AUC']

            self.ROI_Names = []
            
            for index_ROI, ROI in enumerate(range(roi_summaries.shape[0])):
                
                self.ROI_Area.append(roi_summaries.iloc[index_ROI,0])
                self.ROI_Ratio_Peak.append(roi_summaries.iloc[index_ROI,1])
                self.ROI_Peak_Time.append(roi_summaries.iloc[index_ROI,2])
                self.ROI_Ratio_AUC.append(roi_summaries.iloc[index_ROI,3])
                self.ROI_Number.append(index_ROI+1)
                self.ROI_Names.append('ROI_Ref'+str(index_ROI+1))
                
            
            self.ROI_Names.insert(0, "Time [s]")
            ROI_iFlow_data = df.iloc[index_Table[0]+2:-1,0:self.param.ROI+1]
            ROI_iFlow_data.columns = self.ROI_Names
            X = ROI_iFlow_data['Time [s]'].astype(float)
            X2 = sm.add_constant(X)
            
    
            for ROI in ROI_iFlow_data.columns[1:]: 
    
                ROI_iFlow_data[ROI] = ROI_iFlow_data[ROI].astype(float)
                Y = ROI_iFlow_data[ROI]
                
                min_index, min_value = min(enumerate(ROI_iFlow_data[ROI]), key=operator.itemgetter(1))
                max_index, max_value = max(enumerate(ROI_iFlow_data[ROI]), key=operator.itemgetter(1))
            
                ROI_Ref_model_half = sm.OLS(Y.iloc[max_index:],X2.iloc[max_index:,:])
                results_half = ROI_Ref_model_half.fit()
                
                ROI_Ref_model_double = sm.OLS(Y.iloc[:max_index],X2.iloc[:max_index,:])
                results_double = ROI_Ref_model_double.fit()
                
            
                self.halflife.append(-np.log(2) / results_half.params[1])
                self.auc.append(metrics.auc(X, Y))
                
                self.double_time.append(np.log(2)/np.log(1+results_double.params[1]))
                self.elimination_rate.append(np.log(max(Y/min(Y)))/(max(X)- min(X)))
                
                min_index, min_value = min(enumerate(ROI_iFlow_data[ROI]), key=operator.itemgetter(1))
                max_index, max_value = max(enumerate(ROI_iFlow_data[ROI]), key=operator.itemgetter(1))
                
                self.TOMC.append(ROI_iFlow_data['Time [s]'].iloc[max_index])
                self.CMAX.append(max_value)
                self.kurtosis.append(kurtosis(Y))
                self.skewness.append(skew(Y))
                self.ktest.append(kstest(Y,'norm')[0])
                
#            self.halflife_flat= [item for sublist in self.halflife for item in sublist]
#            self.double_time= [item for sublist in self.double_time for item in sublist]
            
            self.patient_name_temp = ' '.join([self.patient_name[:]]*(len(self.ROI_Names)-1))
            self.patient_name_temp = [self.patient_name[:]]*(len(self.ROI_Names)-1)
            self.patient_name_DF.append(self.patient_name_temp)
            self.flat_list_names = [item for sublist in self.patient_name_DF for item in sublist]

            self.patient_mrn_temp = [self.patient_mrn[:]]*(len(self.ROI_Names)-1)

            #self.patient_mrn_DF.append(re.split(' ', self.patient_mrn_temp))
            self.patient_mrn_DF.append(self.patient_mrn_temp)
            self.flat_list_mrn = [item for sublist in self.patient_mrn_DF for item in sublist]
            
              
            self.patient_birthdate_temp = ' '.join([self.patient_birthdate[:]]*(len(self.ROI_Names)-1))
            self.patient_birthdate_DF.append(re.split('  ', self.patient_birthdate_temp))
            self.flat_list_birthdate = [item for sublist in self.patient_birthdate_DF for item in sublist]

            
            self.patient_sex_temp = ' '.join([self.patient_sex[:]]*(len(self.ROI_Names)-1))
            self.patient_sex_DF.append(re.split('  ', self.patient_sex_temp))
            self.flat_list_sex = [item for sublist in self.patient_sex_DF for item in sublist]

            self.procedure_date_temp = [self.procedure_date[:]]*(len(self.ROI_Names)-1)
            self.procedure_date_DF.append(self.procedure_date_temp)
            self.flat_list_procedure_date = [item for sublist in self.procedure_date_DF for item in sublist]

            self.procedure_time_temp = [self.procedure_time[:]]*(len(self.ROI_Names)-1)
            self.procedure_time_DF.append(self.procedure_time_temp)
            self.flat_list_procedure_time = [item for sublist in self.procedure_time_DF for item in sublist]
            
        
        self.iflow_DF = pd.DataFrame(data={'Patient Name':  self.flat_list_names,
                                           'Patient MRN':  self.flat_list_mrn,
                                           'Patient Birthdate': self.flat_list_birthdate, 
                                           'Patient Sex': self.flat_list_sex,
                                           'Procedure Date': self.flat_list_procedure_date ,
                                           'Procedure Time': self.flat_list_procedure_time, 
                                           'ROI Number' : self.ROI_Number, 
                                           'ROI Area' : self.ROI_Area, 
                                           'ROI Ratio Peak' : self.ROI_Ratio_Peak,
                                           'ROI Peak Time' : self.ROI_Peak_Time, 
                                           'ROI Ratio AUC' : self.ROI_Ratio_AUC, 
                                           'Halflife' : self.halflife, 
                                           'Double Time' : self.double_time,
                                           'AUC': self.auc,
                                           'Elimination Rate': self.elimination_rate, 
                                           'Time to Max C': self.TOMC, 
                                           'CMax' : self.CMAX, 
                                           'Kurtosis': self.kurtosis, 
                                           'Skewness': self.skewness,
                                           'D-Value Kulmog': self.ktest
                                           })
    
        self.iflow_DF.to_csv(self.param.out_file+".csv")
    
def main():
    parser = get_parser()
    param = parser.parse_args()
    flow = iFlow_Analysis(param=param)
    flow.processing()
   
if __name__=="__main__":
    main()
        
    