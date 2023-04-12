# -*- coding: utf-8 -*-
import pandas as pd
import torch
import numpy as np
import random

ratings = pd.read_csv("http://www.cs.toronto.edu/~guerzhoy/324/movielens/ratings.csv")
ratings_train = ratings.sort_values(by=['timestamp'])[0:80668]
ratings_test = ratings.sort_values(by=['timestamp'])[80668:-1]

