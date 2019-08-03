import numpy as np
import tensorflow as tf




'''def model(qX, pX, lX):
	query_X = tf.placeholder(tf.float32, [None, timesteps_Q, embed_dims])
	passage_X = tf.placeholder(tf.float32, [None, timesteps_P, embed_dims])
	labels = tf.placeholder(tf.float32, [None])
	Q_input = tf.unstack(query_X, timesteps_Q, 1)
	P_input = tf.unstack(passage_X, timesteps_P, 1)
	W1 = tf.get_variable("W1", [1, timesteps_P], initializer = tf.contrib.layers.xavier_initializer())
	b1 = tf.get_variable("b1", [timesteps_Q], initializer = tf.contrib.layers.xavier_initializer())
	W2 = tf.get_variable("W2", [timesteps_Q, 1], initializer = tf.contrib.layers.xavier_initializer())
	b2 = tf.get_variable("b2", [1], initializer = tf.contrib.layers.xavier_initializer())
	
	lstm_layer_Q = tf.contrib.rnn.LSTMCell(num_units)
	with tf.variable_scope('query'):
		outputs_Q, _ = tf.nn.static_rnn(lstm_layer_Q, Q_input, dtype=tf.float32) #Shape = (timesteps_Q, num_examples, num_units)
	lstm_layer_P = tf.contrib.rnn.LSTMCell(num_units)
	with tf.variable_scope('passage'):
		outputs_P, _ = tf.nn.static_rnn(lstm_layer_P, P_input, dtype=tf.float32) #Shape = (timesteps_P, num_examples, num_units)

	S = tf.math.softmax(tf.tensordot(outputs_P, tf.transpose(outputs_Q), axes=[[1,2],[0,1]]), axis=1)
	Z = tf.add(tf.matmul(W1, S), b1)
	A = tf.math.sigmoid(tf.add(tf.matmul(Z, W2), b2))

	cost = tf.pow(tf.subtract(A, labels), 2)
	optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	sess = tf.Session()
	
	
	sess.run(init)

	for i in range(epochs):
		for j in range(int(num_examples / minibatch_size) + 1):
			#print(j)
			que = qX[minibatch_size * j : minibatch_size * (j + 1)]
			pas = pX[minibatch_size * j : minibatch_size * (j + 1)]
			lab = lX[minibatch_size * j : minibatch_size * (j + 1)]
			_, e_cost = sess.run([optimiser, cost], feed_dict = {query_X : que, passage_X : pas, labels : lab})
			if (j + 1) % 100 == 0:
				print('Cost of batch ' + str(j + 1))
				print(e_cost)
				saver.save(sess, "D:/NNs/RNN/Word embeddings/work/models/model.ckpt")

	save_path = saver.save(sess, "D:/NNs/RNN/Word embeddings/work/models/model.ckpt")
	print("Model saved in path: %s" % save_path)'''

def shuffle(X):
	#np.random.shuffle(X)

	query_X = [X[:,0]]
	passage_X = [X[:,1]]
	labels_X = [X[:,2]]

	query_X = query_X[0][:].tolist()
	passage_X = passage_X[0][:].tolist()
	labels_X = labels_X[0][:].tolist()

	return query_X, passage_X, labels_X


def model(data_file):
	file = open(data_file, 'r', encoding='utf-8')
	#query_X = []
	#passage_X = []
	#labels_X = []
	X = []
	i = 0
	l = 0
	stop = 1

	query_X = tf.placeholder(tf.float32, [None, timesteps_Q, embed_dims])
	passage_X = tf.placeholder(tf.float32, [None, timesteps_P, embed_dims])
	labels = tf.placeholder(tf.float32, [None])
	Q_input = tf.unstack(query_X, timesteps_Q, 1)
	P_input = tf.unstack(passage_X, timesteps_P, 1)
	WA = tf.get_variable("WA", [4 * num_units, 2 * num_units], initializer = tf.contrib.layers.xavier_initializer())
	bA = tf.get_variable("bA", [2 * num_units], initializer = tf.contrib.layers.xavier_initializer())
	W1 = tf.get_variable("W1", [timesteps_P * 2 * num_units, hidden_size_1], initializer = tf.contrib.layers.xavier_initializer())
	b1 = tf.get_variable("b1", [hidden_size_1], initializer = tf.contrib.layers.xavier_initializer())
	W2 = tf.get_variable("W2", [hidden_size_1, hidden_size_2], initializer = tf.contrib.layers.xavier_initializer())
	b2 = tf.get_variable("b2", [hidden_size_2], initializer = tf.contrib.layers.xavier_initializer())
	W3 = tf.get_variable("W3", [hidden_size_2, hidden_size_3], initializer = tf.contrib.layers.xavier_initializer())
	b3 = tf.get_variable("b3", [hidden_size_3], initializer = tf.contrib.layers.xavier_initializer())
	W4 = tf.get_variable("W4", [hidden_size_3, 1], initializer = tf.contrib.layers.xavier_initializer())
	b4 = tf.get_variable("b4", [1], initializer = tf.contrib.layers.xavier_initializer())

	lstm_layer_Q = tf.contrib.rnn.LSTMCell(num_units)
	with tf.variable_scope('query'):
		outputs_Q, _, _ = tf.nn.static_bidirectional_rnn(lstm_layer_Q, lstm_layer_Q, Q_input, dtype=tf.float32) #Shape = (timesteps_Q, num_examples, num_units)
	lstm_layer_P = tf.contrib.rnn.LSTMCell(num_units)
	with tf.variable_scope('passage'):
		outputs_P, _, _ = tf.nn.static_bidirectional_rnn(lstm_layer_P, lstm_layer_P, P_input, dtype=tf.float32) #Shape = (timesteps_P, num_examples, num_units)

	outputs_P_in = tf.transpose(outputs_P, perm=[1,0,2])#tf.reshape(tf.convert_to_tensor(outputs_P),[-1,timesteps_P,2 * num_units])
	outputs_Q_in = tf.transpose(outputs_Q, perm=[1,2,0])#tf.reshape(tf.convert_to_tensor(outputs_Q),[-1,2 * num_units,timesteps_Q])
	S = tf.math.softmax(tf.matmul(outputs_P_in, outputs_Q_in), axis=2)
	C = tf.matmul(S, tf.transpose(outputs_Q, perm=[1,0,2]))
	A_in = tf.concat((C,outputs_P_in),axis=2)
	A = tf.add(tf.tensordot(A_in, WA,[[2],[0]]), bA)
	A_out = tf.reshape(A,[-1,timesteps_P * 2 * num_units])
	#S = tf.matmul(outputs_P_in, outputs_Q_in)
	#S_r = tf.reshape(S,[-1, timesteps_Q * timesteps_P])
	#S = tf.reshape(S,[-1,timesteps_P,timesteps_Q])
	Z1 = tf.add(tf.matmul(A_out, W1), b1)
	A1 = tf.nn.relu(Z1)
	Z2 = tf.add(tf.matmul(A1, W2), b2)
	A2 = tf.nn.relu(Z2)
	Z3 = tf.add(tf.matmul(A2, W3), b3)
	A3 = tf.nn.relu(Z3)
	Z4 = tf.reshape(tf.add(tf.matmul(A3, W4), b4),[-1])
	#Z = tf.reshape(Z,[-1,1,40])
	#A = tf.math.sigmoid(tf.add(tf.tensordot(Z, W2, axes=[[2],[0]]), b2))
	#A = tf.reshape(A,[-1])

	cost = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=Z4))
	optimiser = tf.train.AdamOptimizer().minimize(cost)

	#init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	sess = tf.Session()

	#sess.run(init)
	saver.restore(sess,'D:/NNs/RNN/Word embeddings/work/models14/model.ckpt')
	for ep in range(epochs):
		file = open(data_file, 'r', encoding='utf-8')
		
		X = []
		i = 0
		l = 0

		for example in file:
			if (l + 1) < 1420800:
				if stop:
					print('skipping ' + str(l + 1))
					l = l + 1
					continue

			stop = 0
					
			if l == 2420801:
				break

			tokens = example.strip().lower().split('\t')

			X.append([])
			query_vec = tokens[0]
			passage_vec = tokens[1]
			label = tokens[2]

			query_vec = [float(x) for x in query_vec.split()]
			passage_vec = [float(x) for x in passage_vec.split()]
			label = float(label)

			query_vec = np.array(query_vec).reshape(-1,embed_dims)
			query_vec = np.append(query_vec, np.zeros((timesteps_Q - query_vec.shape[0],50)),axis=0)
			passage_vec = np.array(passage_vec).reshape(-1,embed_dims)
			passage_vec = np.append(passage_vec, np.zeros((timesteps_P - passage_vec.shape[0],50)),axis=0)
			label = np.array(label)

			if label == 1:
				for z in range(3):
					X[i].append(query_vec)
					X[i].append(passage_vec)
					X[i].append(label)
					X.append([])
					i = i + 1	
				
			
			X[i].append(query_vec)
			X[i].append(passage_vec)
			X[i].append(label)
			#query_X.append(query_vec)
			#passage_X.append(passage_vec)
			#labels_X.append(label)

			if (l + 1) % 128 == 0:
				X = np.array(X)
				qX, pX, lX = shuffle(X)
				#print('Example ' + str(l + 1) + ' read')		
				#for k in range(epochs):
				#for j in range(int(40000 / minibatch_size) + 1):
				#print(j)
				que = qX
				pas = pX
				lab = lX
				#lab = np.array(lab).reshape(len(lab),1)
				_, e_cost, e_A = sess.run([optimiser, cost, Z4], feed_dict = {query_X : que, passage_X : pas, labels : lab})
				#e_A = sess.run(A2, feed_dict = {query_X : que, passage_X : pas, labels : lab})
				if (l + 1) % 12800 == 0:
					print('Example : ' + str(l + 1) + '.. Cost of epoch ' + str(ep + 1))
					print(e_cost)
					print(e_A)
					print(lab)
					#print(e_P)
					#print(e_Q)
					#print(lab)
					saver.save(sess, "D:/NNs/RNN/Word embeddings/work/models15/model.ckpt")
				
				#saver.save(sess, "D:/NNs/RNN/Word embeddings/work/models8/model.ckpt")			
				X = []
				i = -1

			i = i + 1	
			l = l + 1	

		saver.save(sess, "D:/NNs/RNN/Word embeddings/work/models15/model.ckpt")
		file.close()
	#query_X = np.array(query_X)
	#passage_X = np.array(passage_X)
	#labels_X = np.array(labels_X)



if __name__ == '__main__':
	
	learning_rate = 0.25
	epochs = 100
	num_units = 64
	num_examples = 5241880
	embed_dims = 50
	num_epochs = 10
	minibatch_size = 128
	timesteps_Q = 40
	timesteps_P = 300
	hidden_size_1 = 1024
	hidden_size_2 = 516
	hidden_size_3 = 256

	data_file = 'evaluate.tsv'
	evaluation_file = 'eval1_unlabelled.tsv'
	result_file = 'answer.tsv'
	

	X = model(data_file)
	print('\nTrained\n')

	#query_X, passage_X, labels_X = shuffle(X)

	#model(query_X, passage_X, labels_X)
	#evaluate(evaluation_file, result_file)