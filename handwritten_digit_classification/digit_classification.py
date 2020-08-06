import random
import numpy as np
import mnist_loader

class Network(object):
	def __init__(self,sizes):
		self.num_layers=len(sizes)
		self.sizes=sizes
		self.biases=[np.random.randn(y,1) for y in sizes[1 :]]
		self.weights = [np.random.randn(y, x) for y, x in zip(sizes[1 :], sizes[: -1])]
	def feedforward(self,a):
		for w,b in zip(self.weights,self.biases):
			a=sigmoid(np.dot(w,a)+b) 
		return a
	def SGD(self, training_data, epochs, mini_batch_size, eta,test_data=None):
		tr_data_len=len(training_data)
		if test_data: test_data_len=len(test_data)
		for i  in xrange(epochs):
			random.shuffle(training_data)
			mini_batches=[training_data[j:j+mini_batch_size] for j in xrange(0,tr_data_len,mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch,eta)
			if test_data:
				print "Epoch {0}: {1}/{2}" .format(i,self.evaluate(test_data),test_data_len)
			else:
				print "Epoch{0}".format(i);
	def update_mini_batch(self,mini_batch,eta):
		nabla_b=[np.zeros(b.shape) for b in self.biases]
		nabla_w=[np.zeros(w.shape) for w in self.weights]
		for (x,y) in mini_batch:
			delta_b,delta_w=self.backprop(x,y)
			nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_b)]
			nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_w)]				
		self.biases = [b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b)]
		self.weights = [w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]
	def backprop(self,x,y):
		nabla_b=[np.zeros(b.shape) for b in self.biases]
		nabla_w=[np.zeros(w.shape) for w in self.weights]
		activation = x
		activations =[x]
		zs = []
		for b,w in zip(self.biases,self.weights):
			z=np.dot(w,activation)+b
			zs.append(z)
			activation=sigmoid(z)
			activations.append(activation)
		delta=(activation-y)*sigmoid_prime(zs[-1])
		nabla_b[-1]=delta
		nabla_w[-1]=np.dot(delta,activations[-2].transpose())
		for l in xrange(2,self.num_layers):
			delta=np.dot(self.weights[-l+1].transpose() , delta)*sigmoid_prime(zs[-l])
			nabla_b[-l]=delta
			nabla_w[-l]=np.dot(delta,activations[-l-1].transpose())
		return (nabla_b,nabla_w)
	def evaluate(self,test_data):
		test_results=[(np.argmax(self.feedforward(x)),y) for (x,y) in test_data]
		return sum(int (x==y) for (x,y) in test_results)
def sigmoid(z):
	return (1.0/(1.0+np.exp(-z)))
def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))


training_data, validation_data, test_data =  mnist_loader.load_data_wrapper()
#print training_data
net = Network([784, 30, 10])
net.SGD(training_data, 30, 10, 10.0, test_data=test_data)