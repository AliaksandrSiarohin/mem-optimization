import lasagne
import lasagne.layers as ll
from util import IMAGE_SHAPE
import theano.tensor as T
import sample_layer

def define_full_net():
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


def define_patch_net():
    net = {}
    print ("Discriminator layer shapes:")
    net['input'] = ll.InputLayer(shape=(None, 3, IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    net['patch'] = sample_layer.Sample2DLayer(net['input'], 50, (32, 32))
    leaky_relu = lasagne.nonlinearities.LeakyRectify(0.2)

    net['conv_1'] = ll.Conv2DLayer(net['patch'], num_filters=16, filter_size=(4, 4),
                                   nonlinearity=leaky_relu)
    print(lasagne.layers.get_output_shape(net['conv_1']))
    net['conv_2'] = ll.batch_norm(ll.Conv2DLayer(net['conv_1'], num_filters=32, filter_size=(4, 4), stride=(2,2),
                                                 nonlinearity=leaky_relu))
    print(lasagne.layers.get_output_shape(net['conv_2']))
    net['conv_3'] = ll.batch_norm(ll.Conv2DLayer(net['conv_2'], num_filters=64, filter_size=(4, 4),
                                                 nonlinearity=leaky_relu))
    print(lasagne.layers.get_output_shape(net['conv_3']))
    net['conv_4'] = ll.batch_norm(ll.Conv2DLayer(net['conv_3'], num_filters=128, stride=(2, 2), filter_size=(4, 4),
                                                 nonlinearity=leaky_relu))
    print(lasagne.layers.get_output_shape(net['conv_4']))
    net['conv_5'] = ll.batch_norm(ll.Conv2DLayer(net['conv_4'], num_filters=256, filter_size=(4, 4),
                                                 nonlinearity=leaky_relu))
    print(lasagne.layers.get_output_shape(net['conv_5']))

    # net['conv_6'] = ll.batch_norm(ll.Conv2DLayer(net['conv_5'], num_filters=512, filter_size=(4, 4), stride = (2, 2),
    #                                              nonlinearity=leaky_relu))
    # print(lasagne.layers.get_output_shape(net['conv_6']))

    net['out'] = ll.DenseLayer(net['conv_5'], num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)

    print(lasagne.layers.get_output_shape(net['out']))

    return net

patch_net = define_patch_net()
full_net = define_full_net()

def define_loss_generator(generated_img):
    patch_out = lasagne.layers.get_output(patch_net['out'], generated_img)
    full_out = lasagne.layers.get_output(full_net['out'], generated_img)

    loss = T.log(patch_out).mean() + T.log(full_out).mean()
    return -loss


def single_discriminator_loss(net, generated_img, input_to_discriminator):
    true = lasagne.layers.get_output(net['out'], inputs = input_to_discriminator)
    generated = lasagne.layers.get_output(net['out'], inputs = generated_img)

    true_loss = -T.log(true).mean()
    generated_loss = -T.log(1 - generated).mean()

    return true_loss + generated_loss


def define_loss_discriminator(generated_img, input_to_discriminator):
    return (single_discriminator_loss(patch_net, generated_img, input_to_discriminator) +
            single_discriminator_loss(full_net, generated_img, input_to_discriminator))


def discriminator_params():
    return (lasagne.layers.get_all_params(patch_net['out'], trainable=True) +
            lasagne.layers.get_all_params(full_net['out'], trainable=True))
