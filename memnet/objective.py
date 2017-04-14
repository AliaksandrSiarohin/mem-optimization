import lasagne
import numpy as np
from lasagne.layers import InputLayer, Conv2DLayer
from lasagne.layers import MaxPool2DLayer, LocalResponseNormalization2DLayer
from lasagne.layers import SliceLayer, concat, DenseLayer
import lasagne.nonlinearities
from util import IMAGE_SHAPE
import sample_layer
#IMAGE_W = 227


def define_net(input_var):
    net = {}
    net['data'] = InputLayer(shape=(None, 3, IMAGE_SHAPE[0], IMAGE_SHAPE[1]), input_var=input_var)

    net['patch'] = sample_layer.Sample2DLayer(net['data'], 5, (227, 227), pad=False)

    # conv1
    net['conv1'] = Conv2DLayer(
        net['patch'],
        num_filters=96,
        filter_size=(11, 11),
        stride=4,
        nonlinearity=lasagne.nonlinearities.rectify)

    # pool1
    net['pool1'] = MaxPool2DLayer(net['conv1'], pool_size=(3, 3), stride=2)

    # norm1
    net['norm1'] = LocalResponseNormalization2DLayer(net['pool1'],
                                                     n=5,
                                                     alpha=0.0001 / 5.0,
                                                     beta=0.75,
                                                     k=1)

    # before conv2 split the data
    net['conv2_data1'] = SliceLayer(net['norm1'], indices=slice(0, 48), axis=1)
    net['conv2_data2'] = SliceLayer(net['norm1'], indices=slice(48, 96), axis=1)

    # now do the convolutions
    net['conv2_part1'] = Conv2DLayer(net['conv2_data1'],
                                     num_filters=128,
                                     filter_size=(5, 5),
                                     pad=2)
    net['conv2_part2'] = Conv2DLayer(net['conv2_data2'],
                                     num_filters=128,
                                     filter_size=(5, 5),
                                     pad=2)

    # now combine
    net['conv2'] = concat((net['conv2_part1'], net['conv2_part2']), axis=1)

    # pool2
    net['pool2'] = MaxPool2DLayer(net['conv2'], pool_size=(3, 3), stride=2)

    # norm2
    net['norm2'] = LocalResponseNormalization2DLayer(net['pool2'],
                                                     n=5,
                                                     alpha=0.0001 / 5.0,
                                                     beta=0.75,
                                                     k=1)

    # conv3
    # no group
    net['conv3'] = Conv2DLayer(net['norm2'],
                               num_filters=384,
                               filter_size=(3, 3),
                               pad=1)

    # conv4
    # group = 2
    net['conv4_data1'] = SliceLayer(net['conv3'], indices=slice(0, 192), axis=1)
    net['conv4_data2'] = SliceLayer(net['conv3'], indices=slice(192, 384), axis=1)
    net['conv4_part1'] = Conv2DLayer(net['conv4_data1'],
                                     num_filters=192,
                                     filter_size=(3, 3),
                                     pad=1)
    net['conv4_part2'] = Conv2DLayer(net['conv4_data2'],
                                     num_filters=192,
                                     filter_size=(3, 3),
                                     pad=1)
    net['conv4'] = concat((net['conv4_part1'], net['conv4_part2']), axis=1)

    # conv5
    # group 2
    net['conv5_data1'] = SliceLayer(net['conv4'], indices=slice(0, 192), axis=1)
    net['conv5_data2'] = SliceLayer(net['conv4'], indices=slice(192, 384), axis=1)
    net['conv5_part1'] = Conv2DLayer(net['conv5_data1'],
                                     num_filters=128,
                                     filter_size=(3, 3),
                                     pad=1)
    net['conv5_part2'] = Conv2DLayer(net['conv5_data2'],
                                     num_filters=128,
                                     filter_size=(3, 3),
                                     pad=1)
    net['conv5'] = concat((net['conv5_part1'], net['conv5_part2']), axis=1)

    # pool 5
    net['pool5'] = MaxPool2DLayer(net['conv5'], pool_size=(3, 3), stride=2)


    # fc6
    net['fc6'] = DenseLayer(
        net['pool5'], num_units=4096,
        nonlinearity=lasagne.nonlinearities.rectify)

    # fc7
    net['fc7'] = DenseLayer(
        net['fc6'],
        num_units=4096,
        nonlinearity=lasagne.nonlinearities.rectify)

    # fc8
    net['out'] = DenseLayer(
        net['fc7'],
        num_units=1,
        nonlinearity=lasagne.nonlinearities.linear)


    # print ('Objective layer shapes:')
    # print (lasagne.layers.get_output_shape(net['pool5']))
    # # fc6
    # net['fc6'] = Conv2DLayer(
    #     net['pool5'], num_filters=4096, filter_size=(6, 6),
    #     nonlinearity=lasagne.nonlinearities.rectify, flip_filters=False)
    # print (lasagne.layers.get_output_shape(net['fc6']))
    # # fc7
    # net['fc7'] = Conv2DLayer(
    #     net['fc6'],
    #     num_filters=4096, filter_size=(1, 1),
    #     nonlinearity=lasagne.nonlinearities.rectify)
    # print (lasagne.layers.get_output_shape(net['fc7']))
    # # fc8
    # net['out'] = Conv2DLayer(
    #     net['fc7'],
    #     num_filters=1, filter_size=(1, 1),
    #     nonlinearity=lasagne.nonlinearities.linear)
    # print (lasagne.layers.get_output_shape(net['out']))
    return net


def load_weights(net, file_name):
    weights = np.load(file_name, encoding='latin1')
    # weights[-2] = weights[-2].T.reshape((1, 4096, 1, 1))
    # weights[-4] = weights[-4].T.reshape((4096, 4096, 1, 1))
    # weights[-6] = weights[-6].T.reshape((4096, 256, 6, 6))
    lasagne.layers.set_all_param_values(net['out'], weights)


def define_loss(input_img, file_name='memnet/external_mem.npy'):
    net = define_net(input_img)
    load_weights(net, file_name)
    output = lasagne.layers.get_output(net['out'], deterministic=True)
    return -output[:, 0]
