import lasagne
import lasagne.layers as ll
from util import IMAGE_SHAPE


def define_net():
    net = {}
    print ("Discriminator layer shapes:")
    net['input'] = ll.InputLayer(shape=(None, 3, IMAGE_SHAPE[0], IMAGE_SHAPE[1]))

    leaky_relu = lasagne.nonlinearities.LeakyRectify(0.2)

    net['conv_1'] = ll.Conv2DLayer(net['input'], num_filters=64, stride=(2, 2), filter_size=(4, 4),
                                   nonlinearity=leaky_relu)
    print(lasagne.layers.get_output_shape(net['conv_1']))
    net['conv_2'] = ll.batch_norm(ll.Conv2DLayer(net['conv_1'], num_filters=128, stride=(2, 2), filter_size=(4, 4),
                                                 nonlinearity=leaky_relu))
    print(lasagne.layers.get_output_shape(net['conv_2']))
    net['conv_3'] = ll.batch_norm(ll.Conv2DLayer(net['conv_2'], num_filters=256, stride=(2, 2), filter_size=(4, 4),
                                                 nonlinearity=leaky_relu))
    print(lasagne.layers.get_output_shape(net['conv_3']))
    net['conv_4'] = ll.batch_norm(ll.Conv2DLayer(net['conv_3'], num_filters=512, stride=(2, 2), filter_size=(4, 4),
                                                 nonlinearity=leaky_relu))
    print(lasagne.layers.get_output_shape(net['conv_4']))
    net['conv_5'] = ll.batch_norm(ll.Conv2DLayer(net['conv_4'], num_filters=512, stride=(2, 2), filter_size=(4, 4),
                                                 nonlinearity=leaky_relu))
    print(lasagne.layers.get_output_shape(net['conv_5']))
    net['conv_6'] = ll.batch_norm(ll.Conv2DLayer(net['conv_5'], num_filters=512, stride=(2, 2), filter_size=(4, 4),
                                                 nonlinearity=leaky_relu))
    print(lasagne.layers.get_output_shape(net['conv_6']))
    net['out'] = ll.DenseLayer(net['conv_6'], num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)

    print(lasagne.layers.get_output_shape(net['out']))

    return net
