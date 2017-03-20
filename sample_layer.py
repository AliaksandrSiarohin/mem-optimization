import theano
from lasagne.layers import Layer
from theano.tensor.shared_randomstreams import RandomStreams


class Sample2DLayer(Layer):
    """
        Sample random patches from 2D input. Result is batch of patches with shape
         (patches_per_example * input_batch_size, num_channels, image_h, image_w).
    """
    def __init__(self, incoming, patches_per_example, patch_size, **kwargs):
        self.rng = RandomStreams()
        super(Sample2DLayer, self).__init__(incoming, **kwargs)
        self.patches_per_example = patches_per_example
        self.patch_size = patch_size

    def get_output_for(self, input, **kwargs):
        def sample_one_image(img, y, x):
            return theano.map(lambda x, y, image: image[:, y:(y+self.patch_size[0]), x:(x+self.patch_size[1])],
                        sequences=[x, y],
                        non_sequences=img)[0]

        y = self.rng.random_integers(size=(input.shape[0], self.patches_per_example), low=0,
                                           high=input.shape[2]-self.patch_size[0])
        x = self.rng.random_integers(size=(input.shape[0], self.patches_per_example), low=0,
                                           high=input.shape[3]-self.patch_size[1])
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


