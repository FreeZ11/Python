import pandas as pd
import numpy as np

data1 = pd.read_csv("ratings.csv",usecols = ["userId","movieId","rating"])
data = data1[data1["movieId"]<10000].to_numpy()

max_user = int(max(data.T[0]))
max_movie = int(max(data.T[1]))
X = np.zeros((max_user,max_movie))
prev_Id = 0
counter = -1

for i in range(len(data)):
	check = int(data[i][0])
	if check != prev_Id:
		counter+=1
	index = int(data[i][1]) - 1
	X[counter][index] = data[i][2]
	prev_Id = check


my_ratings = np.zeros((9018,1))
my_ratings[2570] = 5
my_ratings[31] = 4
my_ratings[259] = 5
my_ratings[1096] = 4

X_norm = np.nan_to_num(X/np.linalg.norm(X,axis=0))


z = np.dot(X_norm, np.nan_to_num(my_ratings/np.linalg.norm(my_ratings)))
z_norm = np.nan_to_num(z/np.linalg.norm(z))

result = np.dot(X_norm.T, z_norm)

helper = []
data1 = pd.read_csv("movies.csv")
data = data1[data1["movieId"]<9018]
for index, row in data.iterrows():
	helper.append((result[int(row.movieId-1)][0],row.title))

sorted_result = sorted(helper,key=lambda x: x[0], reverse=True)

for e in sorted_result:
	print(e[1])

