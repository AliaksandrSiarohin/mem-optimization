import lasagne
import lasagne.layers as ll
from util import IMAGE_SHAPE
import theano.tensor as T

from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers.dnn import batch_norm_dnn as batch_norm

import numpy as np

# def define_patch_net(num_layers=4):
#     net = {}
#     print("Discriminator layer shapes:")
#     net['input'] = ll.InputLayer(shape=(None, 3, IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
#
#     leaky_relu = lasagne.nonlinearities.LeakyRectify(0.2)
#
#     net['conv_1'] = Conv2DLayer(net['input'], num_filters=64, filter_size=(4, 4), stride=(2, 2),
#                                        nonlinearity=leaky_relu)
#     print(lasagne.layers.get_output_shape(net['conv_1']))
#     net['conv_2'] = batch_norm(Conv2DLayer(net['conv_1'], num_filters=128, filter_size=(4, 4), stride=(2, 2),
#                                                  nonlinearity=leaky_relu))
#     print(lasagne.layers.get_output_shape(net['conv_2']))
#     net['conv_3'] = batch_norm(Conv2DLayer(net['conv_2'], num_filters=256, filter_size=(4, 4), stride=(2, 2),
#                                                  nonlinearity=leaky_relu))
#     print(lasagne.layers.get_output_shape(net['conv_3']))
#     net['conv_4'] = batch_norm(Conv2DLayer(net['conv_3'], num_filters=512, filter_size=(4, 4), stride=(2, 2),
#                                                  nonlinearity=leaky_relu))
#     print(lasagne.layers.get_output_shape(net['conv_4']))
#     net['patch'] = Conv2DLayer(net['conv_4'], filter_size=(1, 1),
#                                num_filters=1, nonlinearity=lasagne.nonlinearities.sigmoid)
#     print(lasagne.layers.get_output_shape(net['patch']))
#
#     net['out'] = lasagne.layers.ReshapeLayer(net['patch'], shape=(-1, ))
#
#     print(lasagne.layers.get_output_shape(net['out']))
#
#     return net


import sample_layer

# def define_patch_net(num_layers=4, batch_size = 4):
#     net = {}
#     print ("Discriminator layer shapes:")
#     net['input'] = ll.InputLayer(shape=(None, 3, IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
#     #net['reweight'] = sample_layer.ColorReweightLayer(net['input'])
#
#     net['patch'] = sample_layer.Grid2DLayer(net['input'], IMAGE_SHAPE, (16, 16), (32, 32))
#     leaky_relu = lasagne.nonlinearities.LeakyRectify(0.2)
#
#     net['conv_1'] = Conv2DLayer(net['patch'], num_filters=16, filter_size=(4, 4),
#                                     nonlinearity=leaky_relu)
#     print(lasagne.layers.get_output_shape(net['conv_1']))
#     net['conv_2'] = batch_norm(Conv2DLayer(net['conv_1'], num_filters=32, filter_size=(4, 4), stride=(2, 2),
#                                                  nonlinearity=leaky_relu))
#     print(lasagne.layers.get_output_shape(net['conv_2']))
#     net['conv_3'] = batch_norm(Conv2DLayer(net['conv_2'], num_filters=64, filter_size=(4, 4),
#                                                  nonlinearity=leaky_relu))
#     print(lasagne.layers.get_output_shape(net['conv_3']))
#     net['conv_4'] = batch_norm(Conv2DLayer(net['conv_3'], num_filters=128, filter_size=(4, 4), stride=(2, 2),
#                                                  nonlinearity=leaky_relu))
#     print(lasagne.layers.get_output_shape(net['conv_4']))
#     net['conv_5'] = batch_norm(Conv2DLayer(net['conv_4'], num_filters=256, filter_size=(4, 4),
#                                                  nonlinearity=leaky_relu))
#     print(lasagne.layers.get_output_shape(net['conv_5']))
#
#
#     print (lasagne.layers.get_output_shape(net['conv_5']))
#     net['dence'] = ll.DenseLayer(net['conv_5'], num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
#
#     print(lasagne.layers.get_output_shape(net['dence']))
#
#     net['out'] = lasagne.layers.ReshapeLayer(net['dence'], shape=(batch_size * 2, -1))
#
#     print(lasagne.layers.get_output_shape(net['out']))
#
#     return net


def define_patch_net(num_layers=4):
    net = {}
    print("Discriminator layer shapes:")
    net['input'] = ll.InputLayer(shape=(None, 3, IMAGE_SHAPE[0], IMAGE_SHAPE[1]))

    leaky_relu = lasagne.nonlinearities.LeakyRectify(0.2)

    # net['stand'] = ll.standardize(net['input'], offset=np.array([0, 0, 0], dtype='float32'),
    #                         scale=np.array([128.0, 128.0, 128.0], dtype='float32'))

    prev_layer_name = 'input'

    for i_layer in range(num_layers):
        layer_name = 'conv_%i' % (i_layer + 1)
        if i_layer != 0:
            net[layer_name] = batch_norm(Conv2DLayer(net[prev_layer_name], num_filters=min(512, 64 * (2 ** i_layer)),
                                          filter_size=(4, 4), stride=(2, 2), nonlinearity=leaky_relu))
        else:
            net[layer_name] = batch_norm(Conv2DLayer(net[prev_layer_name], num_filters=64 * (2 ** i_layer),
                                filter_size=(4, 4), stride=(2, 2), nonlinearity=leaky_relu))
        print(lasagne.layers.get_output_shape(net[layer_name]))
        prev_layer_name = layer_name

    net['out'] = batch_norm(Conv2DLayer(net[prev_layer_name], filter_size=(1, 1),
                               num_filters=1, nonlinearity=lasagne.nonlinearities.sigmoid))
    print(lasagne.layers.get_output_shape(net['out']))
    # net['out'] = lasagne.layers.ReshapeLayer(net['patch'], shape=(batch_size * 2, -1))
    #
    # print(lasagne.layers.get_output_shape(net['out']))

    return net


eps = 1e-4
def define_loss_generator(net, generated_img, true_image, loss_type='log'):
    assert(loss_type in ['log', 'sqr'])
    input = T.concatenate([generated_img, true_image], axis=0)
    output = lasagne.layers.get_output(net['out'], inputs=input)
    batch_size = T.int_div(input.shape[0], 2)
    if loss_type == 'sqr':
        loss = (T.sqr(output[:batch_size] - 1)).mean()
    else:
        loss = -T.log(output[:batch_size] + eps).mean()

    return loss
    # patch_out = lasagne.layers.get_output(net['out'], generated_img)
    #
    # loss = -T.log(patch_out).mean()
    # return loss

def define_loss_discriminator(net, generated_img, input_to_discriminator, loss_type='log'):
    assert(loss_type in ['log', 'sqr'])
    input = T.concatenate([generated_img, input_to_discriminator], axis=0)
    output = lasagne.layers.get_output(net['out'], inputs=input)
    batch_size = T.int_div(input.shape[0], 2)
    if loss_type == 'log':
        true_loss = -T.log(output[batch_size:] + eps).mean()
        generated_loss = -T.log(1 - output[:batch_size] + eps).mean()
    else:
        true_loss = (T.sqr(output[batch_size:] - 1)).mean()
        generated_loss = (T.sqr(output[:batch_size])).mean()

    return true_loss + generated_loss


    # true = lasagne.layers.get_output(net['out'], inputs=input)
    # generated = lasagne.layers.get_output(net['out'], inputs=generated_img)
    #
    # true_loss = -T.log(true).mean()
    # generated_loss = -T.log(1 - generated).mean()
    #
    # return true_loss + generated_loss

