import numpy as np
import time
import pickle

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

from keras import backend as K

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

import spectral_decomposition as sd
import RNN_model as rnn


## Parameters for data generation
NUM_CLUSTERS = 4
LEN_LFV = 4				# uaually equals NUM_CLUSTERS


## Parameters for RNN model
TIMESTEP = 4           # the length of input sequence
RATIO_VALIDATION = 0.1 # the ratio of portion of samples as validation set

## RNN Hyperparameters
HIDDEN_SIZE = 10
BATCH_SIZE = 10
EPOCHS = 30

def laplacian_rnn():
	pass
def adjacency_rnn():
	pass
def spectral_embedding_rnn():
	pass
def spectral_embedding_fir_filter():
	pass

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

def link_prediction(predicted_lfv, actual_network):
	print(len(actual_network.edges()))

def plot_loss(i, history):
    ## list all data in history
#     print(history.history.keys())
    ## summarize and draw history for accuracy
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
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

	## raw data
	time_node_lfv = np.array(lfv_sequence)

	## generate the data for RNN model training
	## data is a list of tuples containing (training_samples, validation_samples, testing_sample) for nodes
	## data[node][training;validation:testing].shape
	data = rnn.generate_data(time_node_lfv, TIMESTEP, RATIO_VALIDATION)
	# print(data[0][0].shape, data[0][1].shape, data[0][2].shape)


	predicted_lfv = list()
	for i, node in enumerate(data):
		print("\nNode %d:" %i)

		## model building
		model = rnn.build_model(LEN_LFV, HIDDEN_SIZE, TIMESTEP)

		## model fitting
    	## node[0]: training samples, node[1]: validation samples, node[2]: testing samples
    	# plot_losses = rnn.PlotLosses()	# plot the loss by each iteration.

		history = model.fit(node[0][:, :-1], node[0][:, -1], 
			batch_size=BATCH_SIZE, 
			epochs=EPOCHS, 
			verbose=0,
			# callbacks=[plot_losses], 
			validation_data=(node[1][:, :-1], node[1][:, -1]))
		# plot_loss(i, history)
    
	    ## predict the last latent feature vector for the node
		predicted_lfv.append(model.predict(node[2][:-1].reshape(1, TIMESTEP, len(node[2][-1])), batch_size=1, verbose=0).reshape(len(node[2][-1])))
		# print(predicted_lfv[-1])
		# print(node[2][-1])
	K.clear_session() ## remove the stale model from GPU

	nmi = community_prediction(predicted_lfv, time_node_lfv[-1])
	with open('NMI.txt', 'a') as nmi_file:
		nmi_file.write(str(nmi)+'\n')


	link_prediction(predicted_lfv, network_sequence[-1])

if __name__ == '__main__':
	main()