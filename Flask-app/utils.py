"""
UTILS 
- Helper functions to use for your recommender funcions, etc
- Data: import files/models here e.g.
    - movies: list of movie titles and assigned cluster
    - ratings
    - user_item_matrix
    - item-item matrix 
- Models:
    - nmf_model: trained sklearn NMF model
"""
import pandas as pd
import numpy as np


movie='ml-latest-small/movies.csv'
rating='ml-latest-small/ratings.csv'
movies = pd.read_csv(movie)
movies['movieId']=movies['movieId'].astype(str)
ratings = pd.read_csv(rating)
ratings['movieId']=ratings['movieId'].astype(str)
movies_and_ratings = ratings.join(other=movies.set_index('movieId'), on='movieId',how='left')
movies_and_ratings=movies_and_ratings.pivot_table(index='userId',columns='movieId',values='rating')

movies = pd.read_csv('ml-latest-small/movies.csv')

def movie_to_id(string_titles):
    '''
    converts movie title to id for use in algorithms'''
    
    movieID = movies.set_index('title').loc[string_titles]['movieId']
    movieID = movieID.tolist()
    
    return movieID

def id_to_movie(movieID):
    '''
    converts movie Id to title
    '''
    rec_title = movies.set_index('movieid').loc[movieID]['title']
    
    return rec_title

