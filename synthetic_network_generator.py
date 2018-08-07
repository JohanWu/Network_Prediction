'''Program for generating synthetic network sequence'''

# Author: Tsunghan Wu <tsunghanwu511@gmail.com>
# Last Update: 2018/05/16
# Usage: 
#	python synthetic_network_generator.py


import networkx as nx
import random
import pickle

## Parametes for SBM network generation
NUM_NODES = 100
NUM_CLUSTERS = 4
P_IN = 0.8
P_OUT = 0.1
NUM_MOVING_NODES = 20	# usually 2*NUM_CLUSTERS
FIRST_MOVING_NODE = 1	# ID of the first moving node
TIME_PERIOD = 30000

class SyntheticNetwork_SBM(object):
	'''For generating synthetic network sequence by Stochastic Block Model (SBM).

	Parameters:
	-----------
	num_nodes :
	num_clusters :
	p_in :
	p_out :
	num_moving_nodes :
	first_moving_node :
	time_period :

	'''
	def __init__(self, num_nodes, num_clusters, p_in, p_out, num_moving_nodes, first_moving_node, time_period):
		def select_moving_nodes():
			'''Select moving nodes.
			
			Moving nodes are decided by the designated first moving node, number of moving nodes, and number of all nodes in the network. The moving nodes are selected averagely from the node set.'''
			moving_nodes = [self.first_moving_node]
			interval = self.num_nodes // (self.num_moving_nodes - 1)
			for i in range(self.num_moving_nodes - 1):
				moving_nodes.append((moving_nodes[i] + interval) % self.num_nodes)
			return moving_nodes
		def decide_num_nodes_in_each_community():
			'''Decide how many nodes are in each community.

			Each community has similar numbers of nodes. In other words, the nodes are distributed to all communities averagely.'''
			num_nodes_in_community = list([self.num_nodes // self.num_clusters] * self.num_clusters)
			r = remainder = self.num_nodes % self.num_clusters 
			while remainder > 0:
				num_nodes_in_community[r - remainder] += 1
				remainder -= 1
			return num_nodes_in_community
		self.num_nodes = num_nodes
		self.num_clusters = num_clusters
		self.p_in = p_in
		self.p_out = p_out
		self.num_moving_nodes = num_moving_nodes
		self.first_moving_node = first_moving_node
		self.time_period = time_period
		self.moving_nodes = select_moving_nodes()
		self.num_nodes_in_community = decide_num_nodes_in_each_community()


	## check the nodes and number of links in the network
	def check_links(self, network):
		intra_links, inter_links = 0, 0
		for link in network.edges():
		    if network.node[link[0]]['comm'] == network.node[link[1]]['comm']:
		        intra_links += 1
		    else:
		        inter_links += 1
		print('#intra_links:%d, #inter_links:%d' %(intra_links, inter_links))		

	## assign the community for a node
	def _get_community(self, node_id):
	    for i, n in enumerate(self.num_nodes_in_community):
	        node_id -= n
	        if node_id < 0: return i

	def sbm_one_step(self):
		'''Build the network by Stochastic Block Model. Each moving node migrates to the next community at each time.

		'''
		### generate the first network
		network = nx.Graph()
		## create nodes
		for i in range(self.num_nodes):
			network.add_node(i, comm=self._get_community(i))
		## build links
		for i in range(self.num_nodes):
			for j in range(i+1, self.num_nodes, 1):
				if network.node[i]['comm'] == network.node[j]['comm']: # in the same block
					if random.random() < self.p_in: network.add_edge(i, j)
				else:
					if random.random() < self.p_out: network.add_edge(i, j)
    	## check links
		self.check_links(network)
    	
		network_sequence = list([network.copy()])
		### Simple one-step migration
            
		for t in range(1, self.time_period):
			## decide the next communities for moving nodes
			next_comm = [(network.node[n]['comm'] + 1) % self.num_clusters for n in self.moving_nodes]
			## rewire the links for migrating nodes
			network.remove_nodes_from(self.moving_nodes)
			for i, n in enumerate(self.moving_nodes):
				network.add_node(n, comm=next_comm[i])
			for n in self.moving_nodes:
				for j in network.nodes():
					if n != j:
					    if network.node[n]['comm'] == network.node[j]['comm']: # in the same block
					        if random.random() < self.p_in: network.add_edge(n, j)
					    else:
					        if random.random() < self.p_out: network.add_edge(n, j)
			# self.check_links(network)
			network_sequence.append(network.copy())
			# self.check_links(network_sequence[-1])       
		return network_sequence

	def sbm_varying_step(self):
		'''Build the network by Stochastic Block Model. Each moving node migrates to the community following the rule: j' = ((j + (t mod i) + 1) mod K). j': next community, j: current community, t: time, i: node's ID, K: number of clusters.

		'''
		### generate the first network
		network = nx.Graph()
		## create nodes
		for i in range(self.num_nodes):
			network.add_node(i, comm=self._get_community(i))
		## build links
		for i in range(self.num_nodes):
			for j in range(i+1, self.num_nodes, 1):
				if network.node[i]['comm'] == network.node[j]['comm']: # in the same block
					if random.random() < self.p_in: network.add_edge(i, j)
				else:
					if random.random() < self.p_out: network.add_edge(i, j)
    	## check links
		self.check_links(network)
    	
		network_sequence = list([network.copy()])
		### Simple one-step migration
            
		for t in range(1, self.time_period):
			## decide the next communities for moving nodes
			next_comm = [(network.node[n]['comm'] + (t % n) + 1) % self.num_clusters for n in self.moving_nodes]
			## rewire the links for migrating nodes
			network.remove_nodes_from(self.moving_nodes)
			for i, n in enumerate(self.moving_nodes):
				network.add_node(n, comm=next_comm[i])
			for n in self.moving_nodes:
				for j in network.nodes():
					if n != j:
					    if network.node[n]['comm'] == network.node[j]['comm']: # in the same block
					        if random.random() < self.p_in: network.add_edge(n, j)
					    else:
					        if random.random() < self.p_out: network.add_edge(n, j)
			# self.check_links(network)
			network_sequence.append(network.copy())
			# self.check_links(network_sequence[-1])       
		return network_sequence
    
class SyntheticNetwork_SocialNetwork(object):
	'''For generating synthetic network sequence. The networks has community structures with small world and scale free properties which social networks have.

	'''

if __name__ == "__main__":
	# model = input("Choose network generation model:\n[1]SBM\n[2]Social Network\n:")
	model = '1'

	if model == '1':
		## generate a SBM-based network sequence, return a list of networks
		print('Synthetic Network Generation by SBM')

		## Network Initialization
		network = SyntheticNetwork_SBM(NUM_NODES, NUM_CLUSTERS, P_IN, P_OUT, NUM_MOVING_NODES, FIRST_MOVING_NODE, TIME_PERIOD)
		# print(network.moving_nodes)
		# print(network.num_nodes_in_community)

		# network_sequence = network.sbm_one_step()
		network_sequence = network.sbm_varying_step()
		
	elif model == '2':
		## generate a network sequence which has community structure with Small World and Scale-free properties, return a list of networks
		print('Social Network Generation')

		network = SyntheticNetwork_SocialNetwork()


	# print(len(network_sequence))
	## Write the network sequence into file for reuse.
	pickle.dump(network_sequence, open('network_sequence.p', 'wb'))