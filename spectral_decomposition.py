import numpy as np
import scipy as sp
import numpy.linalg as la
import scipy.sparse as sparse
import networkx as nx

def generate_latent_feature_vector_sequence(network_sequence, len_lfv):
    eigenval_seq = list()
    eigenvec_seq = list()
    lfv_seq = list()
    node_list = [i for i in range(len(network_sequence[0]))]
    for network in network_sequence:     
        val, vec = la.eigh(nx.laplacian_matrix(network, nodelist=node_list).toarray())
        if len(eigenvec_seq) > 0:
            for i in range(len_lfv):
                if np.dot(eigenvec_seq[-1][i], vec[:, i]) < 0:
                    val[i] *= (-1)
                    vec[:, i] *= (-1)
        eigenval_seq.append(val[:len_lfv])
        eigenvec_seq.append([vec[:, i] for i in range(len_lfv)])
        lfv_seq.append([vec[i, :len_lfv] for i in range(len(node_list))])
        
#     print(len(eigenval_seq))
#     print(len(eigenvec_seq))
#     print(len(lfv_seq))
    ## time, node, lfv
    return lfv_seq