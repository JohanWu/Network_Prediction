import numpy as np
from sklearn import linear_model

## regression type
REG = linear_model.Ridge
## regression regularization
alpha = 0.1

def generate_data(tn_lfv_seq, timestep): 
    '''Generate the training examples and validation examples from the entire latent feature vector sequence.

    input shape: (time, node, len_lfv)
    output: a list of (training_samples, validation_samples, testing_samples) for all nodes
    '''
    print("(#period, #node, len_lfv):", tn_lfv_seq.shape)
    nt_lfv_seq = np.swapaxes(tn_lfv_seq, 0, 1)
    print("(#node, #period, len_lfv):", nt_lfv_seq.shape)
    
    data = list()
    for n in range(nt_lfv_seq.shape[0]):
        ### generate training samples, validation samples, and the testing sample for each node
        # print((nt_lfv_seq[n]))
        ## get data samples from the sequence
        ## the length of the subsequence of latent feature vectors is timestep+1. The last vector is the vector for the label.
        samples = list()        
        for t in range(timestep, nt_lfv_seq.shape[1]-1):
            samples.append(nt_lfv_seq[n][t-timestep:t+1])
        samples = np.array(samples)
        # print("shape of samples (#all_samples, timestep+1, len_lfv):", samples.shape)
        # print(samples[-1])
        ## get testing set
        testing = nt_lfv_seq[n][-1-timestep:]
        # print("shape of testing set:", testing.shape)
        # print(testing)

        data.append((samples, testing))

    return data

def fir_filter(X, y):
    model = REG(alpha=alpha).fit(X, y)

    return model