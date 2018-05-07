import numpy as np
import time
import math
import pickle
import operator
import pathlib

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

from keras import backend as K
from keras import layers

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

import spectral_decomposition as sd
import FIR_filter as fir
import RNN_model as rnn


## Parameters for data generation
NUM_CLUSTERS = 4
LEN_LFV = 4				# uaually equals NUM_CLUSTERS


## Parameters for RNN model
TIMESTEP = 4           # the length of input sequence
RATIO_VALIDATION = 0.1 # the ratio of portion of samples as validation set

## RNN Hyperparameters
HIDDEN_SIZE = 20
BATCH_SIZE = 100
EPOCHS = 30

## top-K for link prediction
K_ALL = [i for i in range(100, 5001, 100)]
K_NODE = [i for i in range(5, 101, 5)]

def laplacian_rnn():
	pass

def adjacency_rnn():
	pass

def spectral_embedding_rnn(time_node_lfv, rnn_cell):
	## generate the data for RNN model training
	## data is a list of tuples containing (training_samples, validation_samples, testing_sample) for nodes
	## data[node][training;validation:testing].shape
	data = rnn.generate_data(time_node_lfv, TIMESTEP, RATIO_VALIDATION)
	print(data[0][0].shape, data[0][1].shape, data[0][2].shape)
	# for n in range(len(data)):
	# 	print("print the shape of training samples:")
	# 	print(data[n][0][:,:-1,:].shape)
	# 	print("print the first training sample:")
	# 	print(data[n][0][0])
	# 	print(data[n][0][0,:-1,:])
	# 	print(data[n][0][0,-1,:])
	# 	print("print the shape of validation samples:")
	# 	print(data[n][1].shape)
	# 	print("print the testing sample")
	# 	print(data[n][2])
	# 	print(time_node_lfv[-1][n])
	# 	print(data[n][2][:-1])
	# 	print(data[n][2][:-1, ])
	# 	print(data[n][2][-1])
	# 	print(data[n][2][-1, ])
	# 	input()

	predicted_lfv = list()
	for i, node in enumerate(data):
		print("\nNode %d:" %i)

		## model building
		model = rnn.build_model(LEN_LFV, HIDDEN_SIZE, TIMESTEP, rnn_cell)

		## model fitting
    	## node[0]: training samples, node[1]: validation samples, node[2]: testing samples
    	# plot_losses = rnn.PlotLosses()	# plot the loss by each iteration.

		history = model.fit(node[0][:, :-1, :], node[0][:, -1, :], 
			batch_size=BATCH_SIZE, 
			epochs=EPOCHS, 
			verbose=0,
			# callbacks=[plot_losses], 
			validation_data=(node[1][:, :-1, :], node[1][:, -1, :]))
		# plot_loss(i, history)
    
	    ## predict the last latent feature vector for the node
		predicted_lfv.append(model.predict(node[2][:-1].reshape(1, TIMESTEP, len(node[2][-1])), batch_size=1, verbose=0).reshape(len(node[2][-1])))
		# print(predicted_lfv[-1])
		# print(node[2][-1])
	K.clear_session() ## remove the stale model from GPU
	return predicted_lfv

def spectral_embedding_fir_filter(time_node_lfv):
	data = fir.generate_data(time_node_lfv, TIMESTEP)
	predicted_lfv = list()
	for i, node in enumerate(data):
		print("\nNode %d:" %i)
		predicted_components = list()
		# print(node[1])
		# print(time_node_lfv[-1][i])
		for j in range(LEN_LFV):
			model = fir.fir_filter(node[0][:, :-1, j], node[0][:, -1, j])
			# print(node[1][:-1, j].reshape(1, TIMESTEP))
			predicted_components.append(model.predict(node[1][:-1, j].reshape(1, TIMESTEP))[0])
		# print(predicted_components)
		predicted_lfv.append(predicted_components)
	return predicted_lfv

def community_detection(num_clusters, node_lfv):
    result = KMeans(n_clusters=num_clusters, random_state=0).fit(node_lfv)
    return result.labels_

def community_prediction(predicted_lfv, actual_lfv ):
	print("Clustering by actual latent features vectors:")
	label_actual = community_detection(NUM_CLUSTERS, actual_lfv)
	print(label_actual, '\n')
	print("Clustering by predicted latent feature vectors:")
	label_predicted = community_detection(NUM_CLUSTERS, predicted_lfv)
	print(label_predicted, '\n')

	nmi = normalized_mutual_info_score(label_actual, label_predicted)
	ami = adjusted_mutual_info_score(label_actual, label_predicted)
	print('NMI:', nmi)
	print('AMI:', ami)

	return nmi

def top_k_precision_recall_all(k, sorted_links, actual_network):
	correct = 0
	for i, link in enumerate(sorted_links):
		if i < k:
			if link[0] in actual_network.edges():
				correct += 1
		else:
			break
	# print(correct)

	precision = correct/k
	recall = correct/len(actual_network.edges())

	return precision, recall

def top_k_precision_recall_node(k, sorted_links, actual_network, node):
	## find out the degrees of the node.
	degree = 0
	for link in actual_network.edges():
		if node in link:
			degree += 1
	correct, i = 0, 0
	for link in sorted_links:
		if i < k:
			if node in link[0]:	# link connecting with the node
				# print(link)
				i += 1
				if link[0] in actual_network.edges():
					correct += 1
				# print(i, correct)
				# input()
		else:
			break
	precision = correct/k
	recall = correct/degree

	return precision, recall

def link_prediction(sorted_links, actual_network):
	'''Do link prediction tasks.

	1. link prediction for the top-K links in the network.
	2. link prediction for one stationary node's top-k neighbors. (node 0)
	3. link prediction for one moving node's top-k neighbors. (node 1)
	'''
	precision_all, precision_moving, precision_stationary = list(), list(), list()
	recall_all, recall_moving, recall_stationary = list(), list(), list()

	## link prediction for the whole network
	for k in K_ALL:
		pre, rec = top_k_precision_recall_all(k, sorted_links, actual_network)
		precision_all.append(pre)
		recall_all.append(rec)
	# print('pre_all:', precision_all)
	# print('rec_all:', recall_all)

	## link prediction for the stationary node (node 0) and for the moving node (node 1)
	for k in K_NODE:
		## for the stationary node (node 0)
		pre, rec = top_k_precision_recall_node(k, sorted_links, actual_network, 0)
		precision_stationary.append(pre)
		recall_stationary.append(rec)
		## for the moving node (node 1)
		pre, rec = top_k_precision_recall_node(k, sorted_links, actual_network, 1)
		precision_moving.append(pre)
		recall_moving.append(rec)
	# print('pre_sta:', precision_stationary)
	# print('rec_sta:', recall_stationary)
	# print('pre_mov:', precision_moving)
	# print('rec_mov:', recall_moving)

	return precision_all, recall_all, precision_stationary, recall_stationary, precision_moving, recall_moving

def link_score_lfv(predicted_lfv):
	'''Generate scores for all link paris by computing their cosine similarities based on the latent feature vectors.
	'''
	all_links = dict()
	for i, lfv1 in enumerate(predicted_lfv):
		for j, lfv2 in enumerate(predicted_lfv):
			if i < j:
				all_links[(i, j)] = np.dot(lfv1, lfv2)/(math.sqrt(np.dot(lfv1, lfv1))*math.sqrt(np.dot(lfv2, lfv2)))
	sorted_links = sorted(all_links.items(), key=operator.itemgetter(1), reverse=True)
	return sorted_links

def link_score_cn(previous_network, actual_network):
	'''Generate scores for all link pairs by counting the number of their commmon neighbors.
	'''
	all_links = dict()
	for n1 in previous_network.nodes():
		for n2 in previous_network.nodes():
			if n1 < n2:
				# print(n1, set(previous_network.neighbors(n1)))
				# print(n2, set(previous_network.neighbors(n2)))
				all_links[(n1, n2)] = len(set(previous_network.neighbors(n1)).intersection(set(previous_network.neighbors(n2))))
	sorted_links = sorted(all_links.items(), key=operator.itemgetter(1), reverse=True)
	return sorted_links


def plot_loss(i, history):
    ## list all data in history
    # print(history.history.keys())
    ## summarize and draw history for accuracy
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
	## summarize and draw history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['loss', 'val_loss'], loc='upper right')

	## save training loss and validation loss
	plt.savefig('./loss_figure/{}.png'.format(i))
	plt.close()
	# plt.show()
#     if i in moving_nodes:
#         plt.savefig("%s.png" %i)
#         input()


def main():
	network_sequence = pickle.load(open('network_sequence.p', 'rb'))

	## convert the network sequence to a nodes' latent feature vector sequence (lfv_sequence).
	lfv_sequence = sd.generate_latent_feature_vector_sequence(network_sequence, LEN_LFV)
	print(len(lfv_sequence))
	
	## raw data
	time_node_lfv = np.array(lfv_sequence)

	## creat the folder Results/ if it doesn't exist
	pathlib.Path('./Results/').mkdir(parents=True, exist_ok=True) 

	## different predictive models
	time1 = time.time()
	predicted_lfv_sefir = spectral_embedding_fir_filter(time_node_lfv)
	time2 = time.time()
	predicted_lfv_sernn_gru = spectral_embedding_rnn(time_node_lfv, layers.GRU)
	time3 = time.time()
	predicted_lfv_sernn_lstm = spectral_embedding_rnn(time_node_lfv, layers.LSTM)
	time4 = time.time()
	with open('./Results/Training_time_fir.txt', 'a') as time_file:
		time_file.write(str(time2-time1)+'\n')
	with open('./Results/Training_time_rnn_gru.txt', 'a') as time_file:
		time_file.write(str(time3-time2)+'\n')
	with open('./Results/Training_time_rnn_lstm.txt', 'a') as time_file:
		time_file.write(str(time4-time3)+'\n')

	## community prediction for different predictive models
	nmi_sefir = community_prediction(predicted_lfv_sefir, time_node_lfv[-1])
	with open('./Results/NMI_SEFIR.txt', 'a') as nmi_file:
		nmi_file.write(str(nmi_sefir)+'\n')

	nmi_sernn_gru = community_prediction(predicted_lfv_sernn_gru, time_node_lfv[-1])
	with open('./Results/NMI_SERNN_GRU.txt', 'a') as nmi_file:
		nmi_file.write(str(nmi_sernn_gru)+'\n')

	nmi_sernn_lstm = community_prediction(predicted_lfv_sernn_lstm, time_node_lfv[-1])
	with open('./Results/NMI_SERNN_LSTM.txt', 'a') as nmi_file:
		nmi_file.write(str(nmi_sernn_lstm)+'\n')


	## link prediction for FIR
	sorted_links = link_score_lfv(predicted_lfv_sefir)
	pre_all, rec_all, pre_stationary, rec_stationary, pre_moving, rec_moving = link_prediction(sorted_links, network_sequence[-1])
	with open('./Results/Precision_all_SEFIR.txt', 'a') as f:
		f.write(str(pre_all)+'\n')
	with open('./Results/Recall_all_SEFIR.txt', 'a') as f:
		f.write(str(rec_all)+'\n')
	with open('./Results/Precision_stationary_SEFIR.txt', 'a') as f:
		f.write(str(pre_stationary)+'\n')
	with open('./Results/Recall_stationary_SEFIR.txt', 'a') as f:
		f.write(str(rec_stationary)+'\n')
	with open('./Results/Precision_moving_SEFIR.txt', 'a') as f:
		f.write(str(pre_moving)+'\n')
	with open('./Results/Recall_moving_SEFIR.txt', 'a') as f:
		f.write(str(rec_moving)+'\n')

	## link prediction for RNN (GRU)
	sorted_links = link_score_lfv(predicted_lfv_sernn_gru)
	pre_all, rec_all, pre_moving, rec_moving, pre_stationary, rec_stationary = link_prediction(sorted_links, network_sequence[-1])
	with open('./Results/Precision_all_SERNN_GRU.txt', 'a') as f:
		f.write(str(pre_all)+'\n')
	with open('./Results/Recall_all_SERNN_GRU.txt', 'a') as f:
		f.write(str(rec_all)+'\n')
	with open('./Results/Precision_stationary_SERNN_GRU.txt', 'a') as f:
		f.write(str(pre_stationary)+'\n')
	with open('./Results/Recall_stationary_SERNN_GRU.txt', 'a') as f:
		f.write(str(rec_stationary)+'\n')
	with open('./Results/Precision_moving_SERNN_GRU.txt', 'a') as f:
		f.write(str(pre_moving)+'\n')
	with open('./Results/Recall_moving_SERNN_GRU.txt', 'a') as f:
		f.write(str(rec_moving)+'\n')
	
	## link prediction for RNN (LSTM)
	sorted_links = link_score_lfv(predicted_lfv_sernn_lstm)
	pre_all, rec_all, pre_moving, rec_moving, pre_stationary, rec_stationary = link_prediction(sorted_links, network_sequence[-1])
	with open('./Results/Precision_all_SERNN_LSTM.txt', 'a') as f:
		f.write(str(pre_all)+'\n')
	with open('./Results/Recall_all_SERNN_LSTM.txt', 'a') as f:
		f.write(str(rec_all)+'\n')
	with open('./Results/Precision_stationary_SERNN_LSTM.txt', 'a') as f:
		f.write(str(pre_stationary)+'\n')
	with open('./Results/Recall_stationary_SERNN_LSTM.txt', 'a') as f:
		f.write(str(rec_stationary)+'\n')
	with open('./Results/Precision_moving_SERNN_LSTM.txt', 'a') as f:
		f.write(str(pre_moving)+'\n')
	with open('./Results/Recall_moving_SERNN_LSTM.txt', 'a') as f:
		f.write(str(rec_moving)+'\n')

	## link prediction by the commom neigbors of previous network.
	sorted_links = link_score_cn(network_sequence[-2], network_sequence[-1])
	pre_all, rec_all, pre_moving, rec_moving, pre_stationary, rec_stationary = link_prediction(sorted_links, network_sequence[-1])
	with open('./Results/Precision_all_CN.txt', 'a') as f:
		f.write(str(pre_all)+'\n')
	with open('./Results/Recall_all_CN.txt', 'a') as f:
		f.write(str(rec_all)+'\n')
	with open('./Results/Precision_stationary_CN.txt', 'a') as f:
		f.write(str(pre_stationary)+'\n')
	with open('./Results/Recall_stationary_CN.txt', 'a') as f:
		f.write(str(rec_stationary)+'\n')
	with open('./Results/Precision_moving_CN.txt', 'a') as f:
		f.write(str(pre_moving)+'\n')
	with open('./Results/Recall_moving_CN.txt', 'a') as f:
		f.write(str(rec_moving)+'\n')	

	## link prediction by previous network for the entire network
	link_in_T = 0
	for link in network_sequence[-2].edges():
		if link in network_sequence[-1].edges():
			link_in_T += 1
	# print(link_in_T)
	with open('./Results/Precision_all_Previous.txt', 'a') as f:
		f.write(str(link_in_T/len(network_sequence[-2].edges()))+'\n')
	with open('./Results/Recall_all_Previous.txt', 'a') as f:
		f.write(str(link_in_T/len(network_sequence[-1].edges()))+'\n')

	## link prediction by previous network for the stationary node (node 0)
	link_in_T = 0
	guess_link = 0	# links of node 0 in the network at time T-1
	actual_link = 0	# links of node 0 in the network at time T
	for link in network_sequence[-2].edges():
		if 0 in link:
			# print(link)
			guess_link += 1
			if link in network_sequence[-1].edges():
				link_in_T += 1
	for link in network_sequence[-1].edges():
		if 0 in link:
			actual_link += 1
	with open('./Results/Precision_stationary_Previous.txt', 'a') as f:
		f.write(str(link_in_T/guess_link)+'\n')
	with open('./Results/Recall_stationary_Previous.txt', 'a') as f:
		f.write(str(link_in_T/actual_link)+'\n')

	## link prediction by previous network for the moving node (node 1)
	link_in_T = 0
	guess_link = 0	# links of node 1 in the network at time T-1
	actual_link = 0	# links of node 1 in the network at time T
	for link in network_sequence[-2].edges():
		if 1 in link:
			# print(link)
			guess_link += 1
			if link in network_sequence[-1].edges():
				link_in_T += 1
	for link in network_sequence[-1].edges():
		if 1 in link:
			actual_link += 1
	with open('./Results/Precision_moving_Previous.txt', 'a') as f:
		f.write(str(link_in_T/guess_link)+'\n')
	with open('./Results/Recall_moving_Previous.txt', 'a') as f:
		f.write(str(link_in_T/actual_link)+'\n')

if __name__ == '__main__':
	main()
