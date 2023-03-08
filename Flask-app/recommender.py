"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""

import pandas as pd
import numpy as np
from utils import movies,movies_and_ratings
from recom_system import recommendation_system_NMF
import pickle

def loaded_EMF_model():
        with open('nmf_C58_model.pkl','rb') as file:
            loaded_model = pickle.load(file)
            print("file loaded successfully")          
        return  loaded_model


def recommend_random(k=3):
    return movies['title'].sample(k).to_list()

def recommend_with_NMF(usersID,movies_query_dic,k=3):
    recommendation_system=recommendation_system_NMF(movies_and_ratings)
    loaded_model=loaded_EMF_model()
    New_User=recommendation_system.new_user(usersID,movies_query_dic,loaded_model)
    print("1 successfully") 
    #the_recommendation=recommendation_system.Top_recommendation(k)
    #print("2 successfully") 
    
    return movies['title'].sample(k).to_list()
    """
    NMF Recommender
    INPUT
    - user_vector with shape (1, #number of movies)
    - user_item_matrix
    - trained NMF model

    OUTPUT
    - a list of movieIds
    """
    pass

def recommend_neighborhood(query, model, k=3):
    """
    Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model. 
    Returns a list of k movie ids.
    """   
    pass
    

