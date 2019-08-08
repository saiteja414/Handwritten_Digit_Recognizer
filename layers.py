import numpy as np
import math as mt

class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes):
		# Method to initialize a Fully Connected Layer
		# Parameters
		# in_nodes - number of input nodes of this layer
		# out_nodes - number of output nodes of this layer
		self.in_nodes = in_nodes
		self.out_nodes = out_nodes
		# Stores the outgoing summation of weights * feautres 
		self.data = None

		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))	
		self.biases = np.random.normal(0,0.1, (1, out_nodes))
		###############################################
		# NOTE: You must NOT change the above code but you can add extra variables if necessary 

	def forwardpass(self, X):
		# print('Forward FC ',self.weights.shape)
		# Input
		# activations : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_nodes]
		# OUTPUT activation matrix		:[n X self.out_nodes]
		###############################################
		out = np.zeros((n,self.out_nodes))
		bias = np.tile(self.biases,(n,1))
		out = np.add(np.matmul(X,self.weights), bias)
		self.data = out
		out = np.reciprocal(1+np.exp(-1*out))
		return out
		###############################################
		
	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size
		###############################################
		new_delta = np.sum(np.multiply(np.tile(np.multiply(delta,derivative_sigmoid(self.data)),self.in_nodes).reshape(n,self.in_nodes,self.out_nodes),(np.tile(self.weights,[n,1])).reshape(n,self.in_nodes,self.out_nodes)),2)
		error_weights =np.multiply(np.tile(np.multiply(delta, derivative_sigmoid(self.data)),[1,self.in_nodes]), (np.repeat(activation_prev,self.out_nodes)).reshape(n,self.in_nodes*self.out_nodes))
		error_weights = error_weights.reshape(n,self.in_nodes,self.out_nodes)
		error_bias = np.multiply(delta,derivative_sigmoid(self.data))
		self.weights = np.subtract(self.weights, lr * np.sum(error_weights,0)) 
		self.biases = np.subtract(self.biases, lr * np.sum(error_bias,0)) 
		return new_delta
		###############################################

class ConvolutionLayer:
	def __init__(self, in_channels, filter_size, numfilters, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for convolution layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer
		# numfilters  - number of feature maps (denoting output depth)
		# stride	  - stride to used during convolution forward pass
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = numfilters
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

		# Stores the outgoing summation of weights * feautres 
		self.data = None
		
		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))	
		self.biases = np.random.normal(0,0.1,self.out_depth)
		

	def forwardpass(self, X):
		# print('Forward CN ',self.weights.shape)
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		out = np.zeros([n,self.out_depth, self.out_row, self.out_col])
		self.data = np.zeros([n,self.out_depth,self.out_row,self.out_col])
		for t in range(n):
			for f in range(self.out_depth):
				for i in range(self.out_row):
					for j in range(self.out_col):
						out[t,f,i,j] = np.sum(np.multiply(self.weights[f,:,:,:],X[t,:,(self.stride)*i:(self.stride)*i+self.filter_row,(self.stride)*j:(self.stride)*j+self.filter_col]))
						out[t,f,i,j] += self.biases[f]
						self.data[t,f,i,j] = out[t,f,i,j]
						out[t,f,i,j] = sigmoid(out[t,f,i,j])
		return out
		###############################################

	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		
		new_delta = np.zeros([n,self.in_depth,self.in_row,self.in_col])
		error_weights = np.zeros([n,self.out_depth, self.in_depth, self.filter_row, self.filter_col])
		error_bias = np.zeros(self.out_depth)


		for batch in range(n):
			for t in range(self.out_depth):
				for a in range(self.out_row):
					for b in range(self.out_col):
						derivative = derivative_sigmoid(self.data[batch,t,a,b])
						curr_del = delta[batch,t,a,b]
						mul = (curr_del * derivative) * self.weights[t,:,:,:]
						new_delta[batch,:,a*self.stride : a*self.stride + self.filter_row, b*self.stride : b*self.stride + self.filter_col] += mul 


		for batch in range(n):
			for t in range(self.out_depth):
				for i in range(self.out_row):
					for j in range(self.out_col):
						derivative = derivative_sigmoid(self.data[batch,t,i,j])
						curr_del = delta[batch,t,i,j]
						mul = (curr_del * derivative) * activation_prev[batch,:,i*self.stride:i*self.stride+self.filter_row,j*self.stride:j*self.stride+self.filter_col]
						error_weights[batch,t,:,:,:] += mul 

		error_weights = error_weights.sum(0)

		for t in range(self.out_depth):
			temp1 = np.multiply(delta[:,t,:,:],derivative_sigmoid(delta[:,t,:,:]))
			error_bias[t] = temp1.sum()

		self.weights[:,:,:,:] -= lr * error_weights[:,:,:,:]
		self.biases[:] -= lr * error_bias[:]

		return new_delta


		###############################################
	
class AvgPoolingLayer:
	def __init__(self, in_channels, filter_size, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for max_pooling layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer

		# NOTE: Here we assume filter_size = stride
		# And we will ensure self.filter_size[0] = self.filter_size[1]
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = self.in_depth
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

	def forwardpass(self, X):
		# print('Forward MP ')
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		out = np.zeros((n,self.out_depth, self.out_row, self.out_col))
		for t in range(n):
			for f in range(self.out_depth):
				for i in range(self.out_row):
					for j in range(self.out_col):
						out[t,f,i,j] = np.average(X[t,f,i*self.stride:i*self.stride+self.filter_row,j*self.stride:j*self.stride+self.filter_col])
		return out
		###############################################	


	def backwardpass(self, alpha, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# activations_curr : Activations of current layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		n = activation_prev.shape[0] # batch size

		###############################################
		new_delta = np.zeros([n,self.in_depth,self.in_row,self.in_col])
		# for i in range(self.in_row):
		# 	for j in range(self.in_col):
		# 		low_b = mt.ceil((j - (self.filter_col - 1)) / self.stride)
		# 		low_a = mt.ceil((i - (self.filter_row - 1)) / self.stride)
		# 		high_b = mt.floor(j  / self.stride)
		# 		high_a = mt.floor(i  / self.stride)
		# 		new_delta[:,:,i,j] = np.sum(np.sum(delta[:,:,low_a:high_a+1,low_b:low_b+1],2),2) / (self.filter_row * self.filter_col)

		for batch in range(n):
					for t in range(self.out_depth):
						for a in range(self.out_row):
							for b in range(self.out_col):
								curr_del = delta[batch,t,a,b]
								mul = curr_del / (self.filter_row * self.filter_col) * np.ones(self.in_depth,self.filter_row,self.filter_col)
								new_delta[batch,:,a*self.stride : a*self.stride + self.filter_row, b*self.stride : b*self.stride + self.filter_col] += mul 

		return new_delta
		###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        self.in_batch, self.r, self.c, self.k = X.shape
        return X.reshape(self.in_batch, self.r * self.c * self.k)

    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(self.in_batch, self.r, self.c, self.k)


# Helper Function for the activation and its derivative
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))