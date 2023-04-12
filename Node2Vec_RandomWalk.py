# -*- coding: utf-8 -*-
import random
def Node2vec(starting_node,length, p, q, A):

  path = [starting_node]

  user = True
  # Always start the walk on a user node

  BFS_DFS = "DFS"
  # Default to a DFS walk (this actually doesn't do anything, just need to initalize the variable)

  while len(path) < length:
    # Run while the path is less than the given length 

    current_node = path[-1]
    # Take the last node in the path (the node being explored)

    if user == True:
      neighbours = np.where(A[current_node] == 1)
    else:
      neighbours = np.where(A[:,current_node] == 1)

    # Depending on if the node is a user or movie we need to set its neighbours
    # Users will never be neighbours to other users
    # Movies will never be neighbouts to other movies

    breaker = False
    # Initalizing a breaker variable (to get out of the while loop)

    while True:
      if random.random() < 1/p:
        BFS_DFS = "BFS"
        break 
      for i in range(len(neighbours)):
        if random.random() < 1/q:
          BFS_DFS = "DFS"
          breaker = True
          break
      if breaker:
        break

      # This while loop changes the value of BFS_DFS
      # It sets which type of exploration needs to be done 

    if (len(path)) == 1:
      if (len(neighbours[0])) == 0:
        return None
      else:
        path.append(random.choice(neighbours[0]))

    elif (len(path)) > 1:
      if BFS_DFS == "DFS":
        if len(neighbours[0]) == 0:
          path.append(path[-2])
        else:
          path.append(random.choice(neighbours[0]))      
      elif BFS_DFS == "BFS":
        path.append(path[-2])

    # This simply adds the correct node to the path
    # If DFS add a random neighbour, if BFS go back to the original node
    # If the node has no neighbours go ba
    
    user = not(user)
    # Change the variable from user to movie or vise versa
  return path

def Update_node2vec(n_users, n_movies, embedding):  
  emb_movies = embedding(torch.arange(n_movies).to(device))
  emb_users = embedding(torch.arange(n_movies, n).to(device))

  return emb_movies, emb_users

def Cost_node2vec(n_users, n_movies, emb_users, emb_movies, p, q, walk_length):
  walks = []
  total = 0
  for user in range(n_users):
    walks.append(node2vec(user,walk_length, p, q, A))
    if walks[-1] == None:
      continue
    neighbours_user = walks[user][::2]
    neighbours_movie = walks[user][1::2]

    neigh_user_sum = torch.sum(torch.log(torch.sigmoid(torch.matmul(emb_users[user], torch.transpose(emb_users[neighbours_user],0,1)))))
    neigh_movie_sum = torch.sum(torch.log(torch.sigmoid(torch.matmul(emb_users[user], torch.transpose(emb_movies[neighbours_movie],0,1)))))

    denominator = torch.sum(torch.log(torch.sigmoid(torch.matmul(emb_users[user], torch.transpose(emb_users,0,1))))) + torch.sum(torch.log(torch.sigmoid(torch.matmul(emb_users[user], torch.transpose(emb_movies,0,1)))))

    total += neigh_user_sum + neigh_movie_sum - denominator

  return 1/total