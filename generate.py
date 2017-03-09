import theano
import theano.tensor as T
import lasagne
import argparse
import sys
import numpy as np
import pylab as plt

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment", default = 'mnist',
                        help = "Experiment name, the folder with all network definitions")
    parser.add_argument("--input_image", default='gray.png', help="Input image")
    parser.add_argument("--output_image", default='output.png', help='Output image')
    parser.add_argument("--model", default='mnist/generator.npy', help="Path to generator weights")

    return parser.parse_args()


def compile():
    import generator

    input_to_generator = T.tensor4('img_with_noise', dtype='float32')

    G = generator.define_net()
    generated_img = lasagne.layers.get_output(G['out'], inputs=input_to_generator)

    generate_fn = theano.function([input_to_generator], generated_img, allow_input_downcast=True)

    return generate_fn, G

def main():
    options = parse_args()
    sys.path.insert(0, options.experiment)
    generate_fn, G = compile()
    import util
    lasagne.layers.set_all_param_values(G['out'], np.load(options.model))
    img = plt.imread(options.input_image)
    img = util.preprocess(np.expand_dims(img, axis=0))
    output = generate_fn(util.add_noise(img))
    plt.imsave(options.output_image, np.squeeze(output))

if __name__ == "__main__":
    main()
