import lasagne.layers as ll
import lasagne
from util import IMAGE_SHAPE
import numpy as np

def define_net():
    net = {}

    print ("Generator layer shapes:")
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


    net['unconv_1'] = ll.batch_norm(ll.TransposedConv2DLayer(net['conv_6'], num_filters=512, stride=(2, 2),
                                                filter_size=(5, 5)))
    print(lasagne.layers.get_output_shape(net['unconv_1']))

    concat = ll.ConcatLayer([net['unconv_1'], net['conv_5']], axis=1)
    net['unconv_2'] = ll.batch_norm(ll.TransposedConv2DLayer(concat, num_filters=512, stride=(2, 2),
                                                filter_size=(4, 4)))
    print(lasagne.layers.get_output_shape(net['unconv_2']))

    concat = ll.ConcatLayer([net['unconv_2'], net['conv_4']], axis=1)
    net['unconv_3'] = ll.batch_norm(ll.TransposedConv2DLayer(concat, num_filters=256, stride=(2, 2),
                                                filter_size=(4, 4)))
    print(lasagne.layers.get_output_shape(net['unconv_3']))

    concat = ll.ConcatLayer([net['unconv_3'], net['conv_3']], axis=1)
    net['unconv_4'] = ll.batch_norm(ll.TransposedConv2DLayer(concat, num_filters=128, stride=(2, 2),
                                                filter_size=(5, 5)))
    print(lasagne.layers.get_output_shape(net['unconv_4']))

    concat = ll.ConcatLayer([net['unconv_4'], net['conv_2']], axis=1)
    net['unconv_5'] = ll.batch_norm(ll.TransposedConv2DLayer(concat, num_filters=64, stride=(2, 2),
                                                filter_size=(4, 4)))
    print(lasagne.layers.get_output_shape(net['unconv_5']))

    concat = ll.ConcatLayer([net['unconv_5'], net['conv_1']], axis=1)
    net['unconv_6'] = ll.batch_norm(ll.TransposedConv2DLayer(concat, num_filters=32, stride=(2, 2),
                                                filter_size=(5, 5)))

    print(lasagne.layers.get_output_shape(net['unconv_6']))
    net['pre_out'] = ll.batch_norm(ll.Conv2DLayer(net['unconv_6'], num_filters=3, filter_size=(3,3),
                                                  nonlinearity=lasagne.nonlinearities.tanh, pad='same'))
    print(lasagne.layers.get_output_shape(net['pre_out']))
    net['out'] = ll.standardize(net['pre_out'], offset=np.array([0, 0, 0], dtype='float32'),
                                scale=np.array([1/128.0, 1/128.0, 1/128.0], dtype='float32'))

    print(lasagne.layers.get_output_shape(net['out']))

    return net
