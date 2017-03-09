import lasagne
import numpy as np
import pickle
import skimage.transform
import scipy

import theano
import theano.tensor as T
import matplotlib
matplotlib.use('Agg')

from lasagne.utils import floatX

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

IMAGE_W = 227

def build_memnet():
    from lasagne.layers import InputLayer, Conv2DLayer
    from lasagne.layers import MaxPool2DLayer, LocalResponseNormalization2DLayer
    from lasagne.layers import SliceLayer, concat, DenseLayer
    import lasagne.nonlinearities

    net = {}
    net['data'] = InputLayer(shape=(None, 3, IMAGE_W, IMAGE_W))

    # conv1
    net['conv1'] = Conv2DLayer(
        net['data'],
        num_filters=96,
        filter_size=(11, 11),
        stride = 4,
        nonlinearity=lasagne.nonlinearities.rectify)

    
    # pool1
    net['pool1'] = MaxPool2DLayer(net['conv1'], pool_size=(3, 3), stride=2)

    # norm1
    net['norm1'] = LocalResponseNormalization2DLayer(net['pool1'],
                                                     n=5,
                                                     alpha=0.0001/5.0,
                                                     beta = 0.75,
                                                     k=1)
   
    # before conv2 split the data
    net['conv2_data1'] = SliceLayer(net['norm1'], indices=slice(0, 48), axis=1)
    net['conv2_data2'] = SliceLayer(net['norm1'], indices=slice(48,96), axis=1)

    # now do the convolutions
    net['conv2_part1'] = Conv2DLayer(net['conv2_data1'],
                                     num_filters=128,
                                     filter_size=(5, 5),
                                     pad = 2)
    net['conv2_part2'] = Conv2DLayer(net['conv2_data2'],
                                     num_filters=128,
                                     filter_size=(5,5),
                                     pad = 2)

    # now combine
    net['conv2'] = concat((net['conv2_part1'],net['conv2_part2']),axis=1)
    
    # pool2
    net['pool2'] = MaxPool2DLayer(net['conv2'], pool_size=(3, 3), stride = 2)
    
    # norm2
    net['norm2'] = LocalResponseNormalization2DLayer(net['pool2'],
                                                     n=5,
                                                     alpha=0.0001/5.0,
                                                     beta = 0.75,
                                                     k=1)
    
    # conv3
    # no group
    net['conv3'] = Conv2DLayer(net['norm2'],
                               num_filters=384,
                               filter_size=(3, 3),
                               pad = 1)

    # conv4
    # group = 2
    net['conv4_data1'] = SliceLayer(net['conv3'], indices=slice(0, 192), axis=1)
    net['conv4_data2'] = SliceLayer(net['conv3'], indices=slice(192,384), axis=1)
    net['conv4_part1'] = Conv2DLayer(net['conv4_data1'],
                                     num_filters=192,
                                     filter_size=(3, 3),
                                     pad = 1)
    net['conv4_part2'] = Conv2DLayer(net['conv4_data2'],
                                     num_filters=192,
                                     filter_size=(3,3),
                                     pad = 1)
    net['conv4'] = concat((net['conv4_part1'],net['conv4_part2']),axis=1)
    
    # conv5
    # group 2
    net['conv5_data1'] = SliceLayer(net['conv4'], indices=slice(0, 192), axis=1)
    net['conv5_data2'] = SliceLayer(net['conv4'], indices=slice(192,384), axis=1)
    net['conv5_part1'] = Conv2DLayer(net['conv5_data1'],
                                     num_filters=128,
                                     filter_size=(3, 3),
                                     pad = 1)
    net['conv5_part2'] = Conv2DLayer(net['conv5_data2'],
                                     num_filters=128,
                                     filter_size=(3,3),
                                     pad = 1)
    net['conv5'] = concat((net['conv5_part1'],net['conv5_part2']),axis=1)
    
    # pool 5
    net['pool5'] = MaxPool2DLayer(net['conv5'], pool_size=(3, 3), stride = 2)

    # fc6
    net['fc6'] = DenseLayer(
            net['pool5'],num_units=4096,
            nonlinearity=lasagne.nonlinearities.rectify)

    # fc7
    net['fc7'] = DenseLayer(
        net['fc6'],
        num_units=4096,
        nonlinearity=lasagne.nonlinearities.rectify)

    # fc8
    net['fc8-euclidean'] = DenseLayer(
        net['fc7'],
        num_units = 1,
        nonlinearity=lasagne.nonlinearities.linear)
    
    return net


def build_vgg():
    from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
    from lasagne.layers import Pool2DLayer as PoolLayer
    from lasagne.nonlinearities import softmax
    net = {}
    net['input'] = InputLayer((1, 3, IMAGE_W, IMAGE_W))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2, mode='average_exc_pad')
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2, mode='average_exc_pad')
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_4'], 2, mode='average_exc_pad')
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_4'], 2, mode='average_exc_pad')
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_4'], 2, mode='average_exc_pad')

    return net

MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))
def prep_image(im):
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2)
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (IMAGE_W, w*IMAGE_W/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*IMAGE_W/w, IMAGE_W), preserve_range=True)

    # Central crop
    h, w, _ = im.shape
    im = im[(h//2-IMAGE_W//2):h//2+IMAGE_W//2 + (IMAGE_W % 2 == 1),w//2-IMAGE_W//2:w//2 + IMAGE_W//2 + (IMAGE_W % 2 == 1)]
    
    rawim = np.copy(im).astype('uint8')
    
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    # Convert RGB to BGR
    im = im[::-1, :, :]

    im = im - MEAN_VALUES
    return rawim, floatX(im[np.newaxis])

def deprocess(x):
    x = np.copy(x[0])
    x += MEAN_VALUES

    x = x[::-1]
    x = np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)
    
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_image(photo, mem_coef, content_coef, variation_coef, content_layer, memnet, contentnet, num_iter):
    mem_fun = theano.function([memnet['data'].input_var], lasagne.layers.get_output(memnet['fc8-euclidean']),
                              allow_input_downcast=True)
    print ("Initial memorability %f" % mem_fun(photo))
    def content_loss(P, X, layer):
        p = P[layer]
        x = X[layer]

        loss = 1./2 * ((x - p)**2).mean()
        return loss

    def total_variation_loss(x):
        return (((x[:,:,:-1,:-1] - x[:,:,1:,:-1])**2 + (x[:,:,:-1,:-1] - x[:,:,:-1,1:])**2)**1.25).mean()
    
    layers = [content_layer]
    layers = {k: contentnet[k] for k in layers}
    # Precompute layer activations for photo and artwork
    input_im_theano = T.tensor4()
    outputs = lasagne.layers.get_output(layers.values(), input_im_theano)

    photo_features = {k: theano.shared(output.eval({input_im_theano: photo}))
                      for k, output in zip(layers.keys(), outputs)}

    # Get expressions for layer activations for generated image
    generated_image = theano.shared(floatX(np.random.uniform(-128, 128, (1, 3, IMAGE_W, IMAGE_W))))

    gen_features = lasagne.layers.get_output(layers.values(), generated_image)
    gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}
    # Define loss function
    losses = []

    # content loss
    losses.append(content_coef * content_loss(photo_features, gen_features, content_layer))

    # style loss
    losses.append(-mem_coef * lasagne.layers.get_output(memnet['fc8-euclidean'], generated_image).mean())

    # total variation penalty
    losses.append(variation_coef * total_variation_loss(generated_image))

    total_loss = sum(losses)
    grad = T.grad(total_loss, generated_image)
    # Theano functions to evaluate loss and gradient
    f_loss = theano.function([], total_loss)
    f_grad = theano.function([], grad)
    

    # Helper functions to interface with scipy.optimize
    def eval_loss(x0):
        x0 = floatX(x0.reshape((1, 3, IMAGE_W, IMAGE_W)))
        generated_image.set_value(x0)
        return f_loss().astype('float64')

    def eval_grad(x0):
        x0 = floatX(x0.reshape((1, 3, IMAGE_W, IMAGE_W)))
        generated_image.set_value(x0)
        return np.array(f_grad()).flatten().astype('float64')
    
    # Initialize with a noise image
    generated_image.set_value(floatX(np.random.uniform(-128, 128, (1, 3, IMAGE_W, IMAGE_W))))

    x0 = generated_image.get_value().astype('float64')
    xs = []
    xs.append(x0)

    # Optimize, saving the result periodically
    for i in range(num_iter):
        scipy.optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, maxfun = 100)
        x0 = generated_image.get_value().astype('float64')
        m = mem_fun(x0)
        print("Iteration: %i. Memorability %f" % (i + 1, m.mean()))
        xs.append(x0)
    return xs[-1]

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_img", default = '00035247.jpg', 
                        help = "Image which memorability optimized")
    parser.add_argument("--output_img", default = 'output.jpg',
                        help = "Output image")
    parser.add_argument("--mem_coef", default = 10000, type = float,
                        help = "Weight of memorability loss")
    parser.add_argument("--content_coef", default = 100000, type = float,
                        help = "Weight of content loss")
    parser.add_argument("--variation_coef", default = 50,  type = float, 
                        help = "Weight of variation loss, more make image more smooth")
    parser.add_argument("--content_layer", default = 'conv4_1', help = "layer that is used in content loss")
    parser.add_argument("--num_iter", type = int , default = 10, help = "Number of iterations")
  
    return parser.parse_args()


def main():
    options = parse_args()
    photo = plt.imread(options.input_img)
    rawim, photo = prep_image(photo)
    
    memnet = build_memnet()
    values = np.load('memnet/memnet.npy', encoding='latin1')
    lasagne.layers.set_all_param_values(memnet['fc8-euclidean'], values)
    
    contentnet = build_vgg()
    values = pickle.load(open('contentnet/vgg19_normalized.pkl', 'rb'), encoding = 'latin-1')['param values']
    lasagne.layers.set_all_param_values(contentnet['pool5'], values)
    
    result_img = generate_image(photo, options.mem_coef, options.content_coef, options.variation_coef, options.content_layer, memnet, contentnet, options.num_iter)
    plt.imsave(options.output_img, deprocess(result_img))
    
if __name__ == '__main__':
    main()
