import lasagne.layers as ll
import lasagne


def define_net():
    net = {}
    # net['input_img'] = ll.InputLayer(shape=(None, 1, 28, 28), input_var=input_img)
    # net['input_noise'] = ll.InputLayer(shape=(None, 1, 28, 28), input_var=noise)
    # ll.ConcatLayer([net['input_img'], net['input_noise']], axis=1)
    net['input'] = ll.InputLayer(shape=(None, 2, 28, 28))

    net['conv_1'] = ll.batch_norm(ll.Conv2DLayer(net['input'], num_filters=32, stride=(2, 2), filter_size=(5, 5),
                                                 pad = 'same'))
    net['conv_2'] = ll.batch_norm(ll.Conv2DLayer(net['conv_1'], num_filters=64, stride=(2, 2), filter_size=(5, 5),
                                                 pad = 'same'))
    net['unconv_3'] = ll.batch_norm(ll.TransposedConv2DLayer(net['conv_2'], filter_size=(5, 5),
                                                        num_filters=32, stride=(2, 2), crop=(2,2)))
    net['out'] = ll.batch_norm(ll.TransposedConv2DLayer(net['unconv_3'], filter_size=(4, 4),
                                                             num_filters=1, stride=(2, 2), crop = (0,0),
                                                             nonlinearity=lasagne.nonlinearities.sigmoid))
    return net
