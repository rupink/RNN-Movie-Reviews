# -*- coding: utf-8 -*-
import torch
import numpy as np
def Map_Movies_Users(ratings_train):
    map_movies = {}
    map_users = {}
    map_idx_user = {}
    map_idx_movie = {}
    
    i = 0
    for index in ratings_train.index:
      if ratings_train["movieId"][index] not in map_movies:
        map_movies[ratings_train["movieId"][index]] = i
        map_idx_movie[i] = ratings_train["movieId"][index]
        i+=1
    
    i = 0
    for index in ratings_train.index:
      if ratings_train["userId"][index] not in map_users:
        map_users[ratings_train["userId"][index]] = i
        map_idx_user[i] = ratings_train["userId"][index]
        i+=1
    
    return (map_movies, map_users, map_idx_user, map_idx_movie)

def Create_Adjacency_Matrix(ratings_train,map_movies, map_users,n_movies,n_users):

    A = np.zeros((n_users, n_movies))
    for index in ratings_train.index:
      if ratings_train["rating"][index] == 5.0:
        A[map_users[ratings_train["userId"][index]]][map_movies[ratings_train["movieId"][index]]] = 1
      else:
        A[map_users[ratings_train["userId"][index]]][map_movies[ratings_train["movieId"][index]]] = -1
        
    return A
