from sklearn.decomposition import NMF 
import pandas as pd
import numpy as np
import pickle

class recommendation_system_NMF(): # shoud be data fram with rowes of iD of users and the movies as featsures by useing NMF (P_matrix,Q_matrix)=R matrix
    
    def __init__(self,dataFram):
        self.users=dataFram.index.to_list() # The users is the P matrix
        self.movies=dataFram.columns.to_list()  # The users is the Q matrix
        self.DF_empty_value_mean=dataFram.fillna(dataFram.mean())  # The frist data fram is the R matrix
        
    
    
    def build_Q_P_matrixs(self,number_of_components=20):
        self.number_of_components=number_of_components
        self.nmf_model = NMF(number_of_components)
        self.nmf_model.fit(self.DF_empty_value_mean)
        self.Q_matrix = self.nmf_model.components_
        
        self.Q_matrix_feature_col=self.nmf_model.feature_names_in_
        self.Q_matrix_feature_index=self.nmf_model.get_feature_names_out()
        
        self.Q_matrix_DF=pd.DataFrame(data=self.Q_matrix,
                                    columns=self.Q_matrix_feature_col,
                                    index= self.Q_matrix_feature_index)
        
        self.P_matrix = self.nmf_model.transform(self.DF_empty_value_mean)# from def __init__ (self.DF_empty_value_mean)
        self.P_matrix_DF=pd.DataFrame(data=self.P_matrix,
                                    columns=self.Q_matrix_feature_index,
                                    index= self.users) # from def __init__ (self.users)
        
        self.R_matrix= np.dot(self.P_matrix,self.Q_matrix)
        #self.R_matrix_DF=pd.DataFrame(data= self.R_matrix,
        #                            columns=self.movies, # from def __init__  (self.movies)
        #                           index= self.users) # from def __init__  (self.users)
         
            
        self.Check_R_matrix_Error= self.nmf_model.reconstruction_err_
        
        return  self.Q_matrix_DF.shape,self.P_matrix_DF.shape,self.Check_R_matrix_Error#self.R_matrix_DF.shape
    
    def Save_EMF_model(self,path,RW='wb'):
        self.path=path
        with open(path,mode=RW) as file:
            pickle.dump(self.nmf_model,file)# from def build_Q_P_matrixs (self.nmf_model) 
            print("file saved successfully")
        with open(path,'rb') as file:
            self.loaded_model = pickle.load(file)
            print("file loaded successfully")          
        return  self.loaded_model  
       
   
    def new_user (self,usersID,movies_query_dic,loaded_model):
        self.usersID=usersID
        self.movies_query_dic=movies_query_dic
        self.new_user_dataframe =  pd.DataFrame(data=movies_query_dic,
                                    columns=self.movies,# from def __init__  (self.movies)
                                    index = [ self.usersID])
        
        self.DF_user_empty_value_mean= self.new_user_dataframe.fillna(self.DF_empty_value_mean.mean())#2  # The frist data fram is the R matrix for new user # from def __init__ (self.DF_empty_value_mean)
        
        self.P_new_user_matrix = loaded_model.transform(self.DF_user_empty_value_mean)# from def build_Q_P_matrixs (self.nmf_model) 
        self.P_new_user_matrix_DF=pd.DataFrame(data=self.P_new_user_matrix,
                                    columns=loaded_model.get_feature_names_out(),#self.Q_matrix_feature_index,# from def build_Q_P_matrixs (self.Q_matrix_feature_index) 
                                    index= [ self.usersID])
        
        self.R_new_user_matrix= np.dot(self.P_new_user_matrix,loaded_model.components_)#self.Q_matrix)# from def build_Q_P_matrixs (self.Q_matrix) 
        
        self.R_new_user_matrix_DF=pd.DataFrame(data=self.R_new_user_matrix,
                                    columns=loaded_model.feature_names_in_,#self.Q_matrix_feature_col,# from def build_Q_P_matrixs (self.Q_matrix_feature_col) 
                                    index= [ self.usersID])
                                    
        self.R_new_user_matrix_DF.transpose().sort_values(by=[self.usersID], ascending=False)
        
        return  self.R_new_user_matrix_DF
    
    def Top_recommendation(self,Number_top=5):

        self.R_new_user_matrix_DF.transpose().loc[list( self.movies_query_dic.keys()),:] = 0
        Top_recommendation=self.R_new_user_matrix_DF.transpose().sort_values(by=[self.usersID],ascending=False).head(Number_top) # from def new_user (self.R_new_user_matrix_DF) & def new_user (self.usersID)
        
        return  Top_recommendation.to_list()
        
    