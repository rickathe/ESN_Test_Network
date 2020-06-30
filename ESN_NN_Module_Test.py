import numpy as np
import ESN_Test_Module as ESN
import NN_Test_Module as NN
from job_stream.inline import Work


# Hyper-Parameters for ESN
inodes = 2
rnodes = 128
onodes = 1
sparsity = 0.05
leakage = .8

train_cycles = 4000
test_cycles = 400


# Hyper-Parameters for Neural Net
set_input_nodes = 2
set_hidden_nodes = 20
set_output_nodes = 1
set_learning_rate = 0.01
set_hidden_layers = 5

training_size = 4000
testing_size = 400
epoch = 5000
iters = 1000

# Allows for plotting either training over epochs or a bar graph with minimum 
# mean error. use 'bar' for bar graph and 'plot' for training line graph.
test_type = 'bar'
#test_type = 'plot'

# Select hyperparameter to be analyzed and values to cycle through.
# NOTE: Where these iteract with the NN are currently hard-coded and must be
# changed manually.
test_runs = 16
hyper_p = "HN"
test_name = "combined_plot_test_4"
layers = "5 Hidden Layer"


# Data Generation
# Creates m rows of 2 integer values to act as inputs.
training_data = np.random.uniform(1, 100, (train_cycles, 2))
# Multiplies the 2 initial values together to get a real answer, sets to a
# m x 1 array.
#training_solutions = np.array(np.multiply(training_data[:,0], \
#    training_data[:,1]), ndmin=2)
training_solutions = np.array(np.multiply(training_data[:,0], \
    training_data[:,1]), ndmin=2).T

testing_data = np.random.uniform(1, 100, (test_cycles,2))
testing_solutions = np.array(np.multiply(testing_data[:,0], \
    testing_data[:,1]), ndmin=2).T

# Takes the maxes and mins of training and testing arrays.
data_max = np.maximum(np.max(training_solutions), np.max(testing_solutions))
data_min = np.minimum(np.min(training_data), np.min(testing_data))

# Scales training and validation data to be 0 < x <= 1.
data_train = (training_data - data_min) / (data_max - data_min)
Y_train = (training_solutions - data_min) \
    / (data_max - data_min)

data_test = (testing_data - data_min) / (data_max - data_min)
Y_test = (testing_solutions - data_min) \
    / (data_max - data_min)


with Work(range(10)) as w:
# Work(range(10)) is the number of cores to utilize
    @w.job
    def RunThis(x):
        
        for i in range(iters):
            # Run i iterations of the ESN and then take the mean error.
            
            '''
            # ESN Sample Run
            Yhat = ESN.run_reservoir(inodes, 200, onodes, leakage, sparsity,
                data_train, Y_train, data_test, flag='petermann', edges=2, 
                randomness=0.2, alpha=0)
            Yhat = Yhat * (data_max - data_min) + data_min
            Yhat_mean = np.asarray([np.mean(np.square(testing_solutions-Yhat))])
            with open('reser_petermann_a00.txt', 'a') as cba:
                np.savetxt(cba, Yhat_mean, delimiter=',')

            # NN Sample Run
            validation_mean = NN.run_neural_net(
                set_input_nodes, 64, set_output_nodes, set_learning_rate, 
                set_hidden_layers, iters, epoch, data_train, Y_train, data_test, 
                testing_solutions, data_min, data_max, test_type)
            with open('nn_mse_64h.txt', 'a') as abc:
                np.savetxt(abc, validation_mean, delimiter=',')
            '''
            


