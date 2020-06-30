import networkx as nx
import numpy as np
import math
import random




class Petermann_Network:

    def __init__(self, rnodes, edges, p, alpha):

        self.rnodes = rnodes
        self.edges = edges
        self.p = p
        self.pN = int(p * (rnodes**2))
        self.alpha = alpha

    def MakeRingGraph(self):

        g = nx.Graph()

        if self.edges % 2 == 0:
            for i in range(self.rnodes):
                for z in range(1, int(self.edges / 2) + 1):
                    j = (i + z) % self.rnodes
                    g.add_edge(i, j)
        else:
            raise ValueError("must specify even number of edges per node")
        
        return g


    def CalcDist(self):

        dist_array = np.zeros([self.rnodes, self.rnodes])
        for i in range(self.rnodes):
            for z in range(self.rnodes):
                # Sweeps every element of every row, and computes the shortest
                # distance from each node to every other node.
                if (i - z) <= (self.rnodes - i + z):
                    dist_array[i,z] = math.sqrt((i-z)**2)
                if (self.rnodes - i + z) < (i - z):
                    dist_array[i,z] = math.sqrt((self.rnodes-i+z)**2)
                if (self.rnodes + i - z) < dist_array[i,z]:
                    dist_array[i,z] = math.sqrt((self.rnodes+i-z)**2)

        return dist_array


    def PowerDecay(self, A):
        # Applies inverse power law to distances between nodes.

        for i in range(self.rnodes):
            # Currently designed for edges=2, needs generalized formulation.
            if i < (self.rnodes - 1):
                A[i,i+1] = 0.0
            if i == (self.rnodes - 1):
                A[i,0] = 0.0
            A[i,i-1] = 0.0

            for j in range(self.rnodes):
                ele = A[i,j]
                if ele > 0.0:
                    A[i,j] = ele ** (-self.alpha)        
        
        return A


    def MakeChoices(self, G, W):

        for i in range(self.pN):
            linear_idx = np.random.choice(W.size, p=W.ravel()/float(W.sum()))
            x,y = np.unravel_index(linear_idx, W.shape)
            # Removes used link weight so it picks w/o replacement
            W[x,y] = 0.0
            # Turn on that edge in the main array.
            G[x,y] = 1

        return G
       
def RunEverything(rnodes, edges, p, alpha):
    pt = Petermann_Network(rnodes, edges, p, alpha)
    A = pt.MakeRingGraph()
    A = nx.to_numpy_matrix(A)
    B = pt.CalcDist()
    C = pt.PowerDecay(B)
    D = pt.MakeChoices(A, C)
    
    return D
