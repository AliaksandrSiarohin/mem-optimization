import lasagne
import numpy as np

def define_net(input_var):
	net = {}
	net['input'] = lasagne.layers.InputLayer(shape = (None, 1, 28, 28), input_var = input_var)
	net['conv_1'] = lasagne.layers.Conv2DLayer(net['input'], num_filters = 32, filter_size = (5, 5))
	net['pool_1'] = lasagne.layers.MaxPool2DLayer(net['conv_1'], pool_size = (2,2))
	net['conv_2'] = lasagne.layers.Conv2DLayer(net['pool_1'], num_filters = 64, filter_size = (5,5))
	net['pool_2'] = lasagne.layers.MaxPool2DLayer(net['conv_2'], pool_size = (2,2))

	net['fc3'] = lasagne.layers.DenseLayer(net['pool_2'], num_units = 1024)
	net['dp3'] = lasagne.layers.dropout(net['fc3'])

	net['fc4'] = lasagne.layers.DenseLayer(net['dp3'], num_units = 256)
	net['dp4'] = lasagne.layers.dropout(net['fc4'])
	net['out'] = lasagne.layers.DenseLayer(net['dp4'], num_units = 10, nonlinearity = lasagne.nonlinearities.softmax)

	return net

def load_weights(net, file_name = 'mnist/mnist_nn.npy'):
	weights = np.load(file_name)
	lasagne.layers.set_all_param_values(net['out'], weights)

def define_loss(input_img, digit = 3):
	net = define_net(input_img)
	load_weights(net)
	output = lasagne.layers.get_output(net['out'], deterministic = True)
	return -output[:, digit]
