# -*- coding: utf-8 -*-
import torch
import numpy as np
def Update(n_users, n_movies, embedding, n):
  emb_movies = embedding(torch.arange(n_movies))
  emb_users = embedding(torch.arange(n_movies, n))

  return emb_users, emb_movies

def Cost(n_users, n_movies, emb_users, emb_movies, A):
    total = 0 
    for i in range(len(A)):
      movies_5star = np.where(A[i] == 1)[0]
      movies_4star = np.where(A[i] != 1)[0]

      five_star_sum = torch.sum(torch.log(torch.sigmoid(torch.matmul(emb_users[i],torch.transpose(emb_movies[movies_5star], 0, 1))))) * 200
      four_star_sum = torch.sum(torch.log(torch.sigmoid(torch.matmul(emb_users[i],torch.transpose(emb_movies[movies_4star], 0, 1)))))

      total+= (five_star_sum - four_star_sum)

    
    return 1/total

def Train(epochs, lr, params, optimizer, n_users, n_movies, embedding, A):
    for epoch in range(epochs):
        emb_users, emb_movies = Update(n_users, n_movies, embedding, n_users+n_movies)
        loss = Cost(n_users, n_movies, emb_users, emb_movies, A)
    
        optimizer.zero_grad()           # cleans the gradients
        loss.backward(retain_graph=True)
        optimizer.step()
      
        if epoch % 50 == 0:
          print("epoch = ", epoch, "loss=", loss)