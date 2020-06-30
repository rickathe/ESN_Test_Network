import numpy as np
import networkx as nx
import Petermann_Module as pm



class Reservoir:

    def __init__(self, inodes, rnodes, onodes, leakage, sparsity, flag='rand',
        edges=0, randomness=0, alpha=2):

        self.inodes = inodes
        self.rnodes = rnodes
        self.onodes = onodes
        self.leakage = leakage
        self.sparsity = sparsity
        self.edges = edges
        self.randomness = randomness
        self.alpha = alpha


        self.Win = np.random.rand(inodes+1, rnodes)

        if flag == 'rand':
            # +1 for Win bias. -0.5 sets weights between -0.5 and 0.5.
            
            self.W = np.random.rand(rnodes, rnodes)

            # Generates an array between 0 and 1. All those values greater than
            # the set sparsity in the array are set to 0 in the original W array.
            self.W[np.random.rand(rnodes, rnodes)>self.sparsity] = 0
        
        if flag == 'watts':
            self.G = nx.watts_strogatz_graph(n = rnodes, k = self.edges, 
                p = self.randomness)
            self.W = nx.to_numpy_matrix(self.G)

        if flag == 'scale':
            self.G = nx.scale_free_graph(n=rnodes, alpha=.2, beta=.6, gamma=.2)
            self.W = nx.to_numpy_matrix(self.G)

        if flag == 'petermann':
            self.W =  pm.RunEverything(self.rnodes, self.edges, self.randomness,
                self.alpha)

        # Takes the maximum absolute value of the eigenvalue of W as the
        # spectral radius, then scales W to have a specral radius slightly
        # less than 1.
        spec_rad = max(abs(np.linalg.eig(self.W)[0]))
        self.W /= spec_rad/0.9


    def reservoir(self, data, new_start=True):
        """
        Arguments:
            data: time-series input [cycles x inodes]
            new_start: whether a new reservoir state should be created
        Matrices:
            dm: design matrix with: bias+input+
                reservoir activity [cycles x (1+inodes+rnodes)]
            R: reservoir activation [1 x rnodes]
        """

        self.dm = np.zeros((data.shape[0], 1+self.inodes+self.rnodes))

        # If false previous R values are used, useful for validation.
        if new_start:
            self.R = np.ones((1, self.rnodes))

        for t in range(data.shape[0]):

            u = data[t] # initialize input at timestep t

            # generate new reservoir state
            # first summand: influence of last reservoir state (same neuron)
            # second summand:
            #   first dot product: influence of input
            #   second dot product: influence of of last reservoir
            #                       state (other neurons)
            # hstack concatenates 1 and input horizontally
            self.R = (1 - self.leakage) * self.R \
                + self.leakage * np.tanh(np.dot(np.hstack((1,u)), self.Win) \
                + np.dot(self.R, self.W))

            # put bias, input & reservoir activation into one row
            self.dm[t] = np.append(np.append(1,u), self.R)

        return self.dm


def run_reservoir(inodes, rnodes, onodes, leakage, sparsity, data_train,
    Y_train, data_test, flag='rand', edges=0, randomness=0, alpha=2):
    
    # Reservoir & Training
    Echo = Reservoir(inodes, rnodes, onodes, leakage, sparsity, flag, edges,
        randomness, alpha)
    # get reservoir activations for training data
    RA_Train = Echo.reservoir(data_train)
    # caclulate output matrix via moore pensore pseudoinverse (linear reg)
    Wout = np.dot(np.linalg.pinv(RA_Train), Y_train )
    # get reservoir activation for test data
    RA_Test = Echo.reservoir(data_test, new_start=False)
    # calculate predictions using output matrix
    Yhat = np.dot(RA_Test, Wout)
    
    return Yhat
    
