import theano
import lasagne
from lasagne.layers import Layer
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as T
from theano.tensor.nnet.neighbours import images2neibs
import numpy as np


class ColorReweightLayer(Layer):
    def __init__(self, incoming, **kwargs):
        self.rng = RandomStreams()
        super(ColorReweightLayer, self).__init__(incoming, **kwargs)
        # self.low = low
        # self.high = high

    def get_output_for(self, input, **kwargs):
        weights = self.rng.uniform(size=(input.shape[0], input.shape[1]))
        #weights = 3 * weights / T.sum(weights, axis=1).reshape((input.shape[0], 1))
        output = input * weights.reshape((input.shape[0], input.shape[1], 1, 1))
        # output = T.minimum(output, self.high)
        # output = T.maximum(output, self.low)
        return output

class Grid2DLayer(Layer):
    def __init__(self, incoming, incoming_shape, step, patch_size, **kwargs):
        self.rng = RandomStreams()
        super(Grid2DLayer, self).__init__(incoming, **kwargs)
        self.step = step
        self.patch_size = patch_size

        y = np.arange(0, incoming_shape[0] - self.step[0], self.step[0])
        x = np.arange(0, incoming_shape[1] - self.step[1], self.step[1])
        y, x = np.meshgrid(x, y)
        y = y.reshape((-1, ))
        x = x.reshape((-1, ))


        self.x = theano.shared(x, allow_downcast=True)
        self.y = theano.shared(y, allow_downcast=True)


    def get_output_for(self, input, **kwargs):
        def sample_one_image(img, y, x):
            return theano.map(lambda x, y, image: image[:, y:(y+self.patch_size[0]), x:(x+self.patch_size[1])],
                        sequences=[x, y],
                        non_sequences=img)[0]

        y = T.repeat(self.y.reshape((1, -1)), axis=1, repeats=input.shape[0])
        x = T.repeat(self.x.reshape((1, -1)), axis=1, repeats=input.shape[0])

        return theano.map(sample_one_image, sequences=[input, y, x])[0].reshape(
                                                (-1, input.shape[1], self.patch_size[0], self.patch_size[1]))


    def get_output_shape_for(self, input_shape):
        return (None, input_shape[1], self.patch_size[0], self.patch_size[1])



class Sample2DLayer(Layer):
    """
        Sample random patches from 2D input. Result is batch of patches with shape
         (patches_per_example * input_batch_size, num_channels, image_h, image_w).
    """
    def __init__(self, incoming, patches_per_example, patch_size, pad=True,  **kwargs):
        self.rng = RandomStreams()
        super(Sample2DLayer, self).__init__(incoming, **kwargs)
        self.patch_size = patch_size
        self.patches_per_example = patches_per_example
        self.pad = pad

    def get_output_for(self, input, **kwargs):
        def sample_one_image(img, y, x):
            return theano.map(lambda x, y, image: image[:, y:(y+self.patch_size[0]), x:(x+self.patch_size[1])],
                        sequences=[x, y],
                        non_sequences=img)[0]
        if self.pad:
            shp = (input.shape[0], input.shape[1], input.shape[2] + self.patch_size[0] * 2 - 2,
                                                   input.shape[3] + self.patch_size[1] * 2 - 2)

            padded_input = T.zeros(shp)
            padded_input = T.set_subtensor(padded_input[:,:, (self.patch_size[0] - 1):(-self.patch_size[0] + 1),
                                               (self.patch_size[1] - 1):(-self.patch_size[1] + 1)], input)

            input = padded_input

        y = self.rng.random_integers(size=(input.shape[0], self.patches_per_example), low=0,
                                           high=input.shape[2] - self.patch_size[0])
        x = self.rng.random_integers(size=(input.shape[0], self.patches_per_example), low=0,
                                           high=input.shape[3] - self.patch_size[1])

        return theano.map(sample_one_image, sequences=[input, y, x])[0].reshape(
                                                (-1, input.shape[1], self.patch_size[0], self.patch_size[1]))


    def get_output_shape_for(self, input_shape):
         if input_shape[0] is None:
            return (None, input_shape[1], self.patch_size[0], self.patch_size[1])
         else:
             return (input_shape[0] * self.patches_per_example, input_shape[1], self.patch_size[0], self.patch_size[1])


class AggregateSamplesLayer(Layer):
    """
        Take the mean accross patches that belongs to same input image.
    """
    def __init__(self, incoming, patches_per_example, **kwargs):
        super(AggregateSamplesLayer, self).__init__(incoming, **kwargs)
        self.patches_per_example = patches_per_example

    def get_output_for(self, input, **kwargs):
        reshaped_input = input.reshape((-1, self.patches_per_example, input.shape[1], input.shape[2], input.shape[3]))
        return theano.tensor.mean(reshaped_input, axis=1)

    def get_output_shape_for(self, input_shape):
        if input_shape[0] is None:
            return (None, input_shape[1], input_shape[2], input_shape[3])
        else:
            return (input_shape[0] / self.patches_per_example, input_shape[1], input_shape[2], input_shape[3])

# def colorize(image):
#     from skimage import color
#     """Return image tinted by the given hue based on a grayscale image."""
#     hsv = color.rgb2hsv(image)
#     hsv[:, :, 0] = np.random.uniform(0, 1, (1, ))
#     #hsv[:, :, 1] = 1  # Turn up the saturation; we want the color to pop!
#     return color.hsv2rgb(hsv)

def colorize(image):
    from skimage import img_as_ubyte, img_as_float
    return img_as_ubyte(img_as_float(image) * np.random.uniform(0, 1, size=(1, 1, 3)))


if __name__ == "__main__":

    import pylab as plt
    # X = T.tensor4(dtype='float32')
    img = plt.imread('datasets/flowers/image_00001.jpg')
    # img = np.expand_dims(np.moveaxis(img, -1, 0), 0)
    # net = lasagne.layers.InputLayer((None, 3, None, None))
    # net = ColorReweightLayer(net)#, 0, 256)
    # fn = theano.function([X], lasagne.layers.get_output(net, inputs=X), allow_input_downcast=True)
    # plt.imsave("kek.jpg", np.moveaxis(np.squeeze(fn(img)), 0, -1).astype('uint8'))
    img = plt.imread('datasets/flowers/image_00001.jpg')
    plt.imsave("kek.jpg", colorize(img))
    plt.imsave("kek-1.jpg", colorize(img))
    plt.imsave("kek-2.jpg", colorize(img))
    #plt.imsave("kek-1.jpg", colorize(img))
    # net = lasagne.layers.InputLayer((None, 1, 256, 256))
    # net = Grid2DLayer(net, (3, 3), (1, 1), (2, 2))
    # import numpy as np
    #
    # a = np.repeat(np.arange(1,  10).reshape((1, 3, 3)), 3, axis=0).reshape((1, 3, 3, 3))
    # #a = np.arange(1, 10).reshape(1, 1, 3, 3)
    # #print (a)
    # fn = theano.function([X], lasagne.layers.get_output(net, inputs=X), allow_input_downcast=True)
    # b = fn(a)
    # #print (b.reshape(3, -1, 4))
    # print (b)

