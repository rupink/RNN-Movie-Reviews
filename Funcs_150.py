# -*- coding: utf-8 -*-
import Imports
def Top150(emb_users, emb_movies):
    all_comb = torch.matmul(emb_users, torch.transpose(emb_movies, 0 , 1))
    # Multiplies the user and movies embeddings matricies together
    
    all_recall = []
    
    for users in range(n_users):
      ind = torch.where(A[users] == 0)[0] 
      # Indicies for the movies that the user has not seen
    
      user_mul_movie = all_comb[users, ind]
      # Gets the dot product values of all the movies that have not been seen for the user 
    
      indicies_sorted = torch.argsort(user_mul_movie,-1,True)
      # Sorts these dot products
    
      if len(indicies_sorted) < 150:
        take_top = len(indicies_sorted)
      else:
        take_top = 150
      # These statements check if there are 150 reccomendations for the movies or less
    
      sorted_movie_ids = ind[indicies_sorted]
      top_150 = sorted_movie_ids[0:take_top]
    
      all_recall.append(top_150)
      # Puts the top movie reccomendations in a list where the indicies represent the user
      
      return all_recall, top_150

def Recall150():
    Recall_150 = {}
    for user in range(n_users):
        indicies_movies = torch.where(A_test[user] == 1)[0]
        real_Ru = set()
        real_Pu = set()
        if len(indicies_movies) == 0:
          Recall_150[user+1] = (None, None)
        else:
      
          for elem in indicies_movies:
            real_Pu.add(map_idx_movie[elem.item()])
          
          for elem in all_recall[user]:
            real_Ru.add(map_idx_movie[elem.item()])
      
          Recall_150[map_idx_user[user]] = (real_Pu & real_Ru, len(real_Pu & real_Ru)/len(real_Pu))
          
    return Recall_150