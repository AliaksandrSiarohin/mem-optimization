import lasagne.layers as ll
import lasagne
from util import IMAGE_SHAPE


def define_net():
    net = {}
    net['input'] = ll.InputLayer(shape=(None, 6, IMAGE_SHAPE[0], IMAGE_SHAPE[1]))

    leaky_relu = lasagne.nonlinearities.LeakyRectify(0.2)

    net['conv_1'] = ll.Conv2DLayer(net['input'], num_filters=64, stride=(2, 2), filter_size=(4, 4),
                                                 nonlinearity=leaky_relu)
    net['conv_2'] = ll.batch_norm(ll.Conv2DLayer(net['conv_1'], num_filters=128, stride=(2, 2), filter_size=(4, 4),
                                                 nonlinearity=leaky_relu))
    net['conv_3'] = ll.batch_norm(ll.Conv2DLayer(net['conv_2'], num_filters=256, stride=(2, 2), filter_size=(4, 4),
                                                 nonlinearity=leaky_relu))
    net['conv_4'] = ll.batch_norm(ll.Conv2DLayer(net['conv_3'], num_filters=512, stride=(2, 2), filter_size=(4, 4),
                                                 nonlinearity=leaky_relu))
    net['conv_5'] = ll.batch_norm(ll.Conv2DLayer(net['conv_4'], num_filters=512, stride=(2, 2), filter_size=(4, 4),
                                                 nonlinearity=leaky_relu))
    net['conv_6'] = ll.batch_norm(ll.Conv2DLayer(net['conv_5'], num_filters=512, stride=(2, 2), filter_size=(4, 4),
                                                 nonlinearity=leaky_relu))
    net['conv_7'] = ll.batch_norm(ll.Conv2DLayer(net['conv_6'], num_filters=512, stride=(2, 2), filter_size=(4, 4),
                                                 nonlinearity=leaky_relu))
    net['conv_8'] = ll.batch_norm(ll.Conv2DLayer(net['conv_7'], num_filters=512, stride=(2, 2), filter_size=(4, 4),
                                                 nonlinearity=leaky_relu))

    net['unconv_9'] = ll.batch_norm(ll.TransposedConv2DLayer(net['conv_8'], num_filters=512, stride=(2, 2),
                                                filter_size=(4, 4)))
    net['unconv_10'] = ll.batch_norm(ll.TransposedConv2DLayer(net['unconv_9'], num_filters=512, stride=(2, 2),
                                                filter_size=(4, 4)))
    net['unconv_11'] = ll.batch_norm(ll.TransposedConv2DLayer(net['unconv_10'], num_filters=512, stride=(2, 2),
                                                filter_size=(4, 4)))
    net['unconv_12'] = ll.batch_norm(ll.TransposedConv2DLayer(net['unconv_11'], num_filters=512, stride=(2, 2),
                                                filter_size=(4, 4)))
    net['unconv_13'] = ll.batch_norm(ll.TransposedConv2DLayer(net['unconv_12'], num_filters=512, stride=(2, 2),
                                                filter_size=(4, 4)))
    net['unconv_14'] = ll.batch_norm(ll.TransposedConv2DLayer(net['unconv_13'], num_filters=256, stride=(2, 2),
                                                filter_size=(4, 4)))
    net['unconv_15'] = ll.batch_norm(ll.TransposedConv2DLayer(net['unconv_14'], num_filters=128, stride=(2, 2),
                                                filter_size=(4, 4)))
    net['unconv_16'] = ll.batch_norm(ll.TransposedConv2DLayer(net['unconv_15'], num_filters=64, stride=(2, 2),
                                                filter_size=(4, 4)))

    net['img'] = ll.TransposedConv2DLayer(net['unconv_15'], num_filters = 3, stride=(2, 2), filter_size=(4, 4))
    net['out'] = ll.ScaleLayer(net['img'], offset=0.5, scales = 1/128.0)

    return net
