import numpy as np
import math
n_steps = 30
n_features = 1

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return X, y


def sequence(df, n):
	seq_aux = df.iloc[:, n].tolist()

	seq = []
	for elem in seq_aux:
		if elem != 0.0 and not np.isnan(elem):
			seq.append(elem)
	
	x, y = split_sequence(seq, n_steps)
	
	x = np.array(x)
	y = np.array(y)

	x = x.reshape((x.shape[0], x.shape[1], n_features))

	return x, y

'''
def sequence2(df, n):
	seq_aux = df.iloc[:, n].tolist()

	seq = []
	for elem in seq_aux:
		if np.isnan(elem):
				seq.append(0.0)
		else:
			seq.append(elem)
	
	x, y = split_sequence(seq, n_steps)
	
	x = np.array(x)
	y = np.array(y)

	x = x.reshape((x.shape[0], x.shape[1], n_features))

	return x, y
'''


def sequence_many(df, n1, n2):
	x = []
	y = []
	
	for i in range(n1, n2):
		seq_aux = df.iloc[:, i].tolist()

		seq = []
		for elem in seq_aux:
			if elem != 0.0 and not np.isnan(elem):
				seq.append(elem)

		seq_x, seq_y = split_sequence(seq, n_steps)

		x += seq_x
		y += seq_y
	
	x = np.array(x)
	y = np.array(y)

	x = x.reshape((x.shape[0], x.shape[1], n_features))

	return x, y




def distance(c1, c2):
  x = ( c1[0] - c2[0] ) ** 2
  y = ( c1[1] - c2[1] ) ** 2
  return math.sqrt( x + y )
  

def adjacency(data, n):
	ad_matrix = []
	neighbors_list = {}

	for wind in data:
		neighbors = []
		nodes = [ (x, data[x]['coordinates']) for x in data] 

		for i in range(n+1):

			closest = nodes[0]
			dist_closest = distance(data[wind]['coordinates'], nodes[0][1])

			for nd in nodes:
				dist = distance( data[wind]['coordinates'], nd[1])
				if dist < dist_closest:
					dist_closest = dist
					closest = nd

			if wind != closest[0]:
				neighbors.append(closest[0])
				
			nodes.remove(closest)

		neighbors_list.update({ wind: np.array(neighbors)})
		ad_matrix.append( [1 if x in neighbors else 0 for x in data] )

	return np.array(ad_matrix), neighbors_list


def adjacency_radius(data, radius):
	ad_matrix = []
	pos = []
	neighbors_list = {}

	nodes = [ (x, data[x]['coordinates']) for x in data] 

	for wind in data:
		pos.append(data[wind]['coordinates'])
		neighbors = []
		found = False

		for nd in nodes:
			dist = distance( data[wind]['coordinates'], nd[1])
			if dist < radius and wind != nd[0]:
				found = True
				neighbors.append(nd[0])
			
		if found == False:
			closest = nodes[0]
			dist_closest = distance(data[wind]['coordinates'], nodes[0][1])
	 
			for nd in nodes:
				dist = distance( data[wind]['coordinates'], nd[1])
				if dist < dist_closest and wind != nd[0]:
						closest = nd[0]
						dist_closest = dist
			neighbors.append(closest)


		neighbors_list.update({ wind: neighbors})
		ad_matrix.append( [1 if x in neighbors else 0 for x in data] )

	return np.array(ad_matrix), neighbors_list, pos




def wind_data(df, name, num_train):

	seq_aux = df[name]

	seq = []
	for elem in seq_aux:
		if np.isnan(elem):
				seq.append(0.0)
		else:
			seq.append(elem)
	
	seq_x, seq_y = split_sequence(seq, n_steps)
	
	x = []
	y = []
	zero = [ 0 for i in range(n_steps)]
	#print(seq_x)
	#print(zero)
	for i in range(len(seq_x)):
		if not np.array_equal(seq_x[i], zero):
			x.append(seq_x[i])
			y.append(seq_y[i])
		#else:
			#print('EO')

	x = np.array(x)
	y = np.array(y)

	x = x.reshape((x.shape[0], x.shape[1], n_features))

	limit = int(len(x) * num_train / 100)
	x_train = x[:limit]
	y_train = y[:limit]

	x_test = x[limit:]
	y_test = y[limit:]

	return x_train, y_train, x_test, y_test

