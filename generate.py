import matplotlib
matplotlib.use('Agg')
import theano
import theano.tensor as T
import lasagne
import argparse
import sys
import numpy as np
import pylab as plt
import os

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment", default = 'memnet',
                        help = "Experiment name, the folder with all network definitions")
    parser.add_argument("--input_folder", default='datasets/flowers',
                        help="Input images")
    parser.add_argument("--output_folder", default='output-39-test-46', help='Output images')
    parser.add_argument("--model", default='memnet/experiment-39/model/generator-46.npy', help="Path to generator weights")
    parser.add_argument("--objective_model", default='memnet/internal_mem.npy', help='Path to objective model weights')
    parser.add_argument("--device", default='cpu', help='Which device to use')

    return parser.parse_args()


def compile(options):
    import generator
    import objective

    input_to_generator = T.tensor4('img_with_noise', dtype='float32')

    G = generator.define_net()
    generated_img = lasagne.layers.get_output(G['out'], inputs=input_to_generator, determenistic = True)
    generate_fn = theano.function([input_to_generator], generated_img, allow_input_downcast=True)
    lasagne.layers.set_all_param_values(G['out'], np.load(options.model))

    input_image = T.tensor4('input_img', dtype='float32')
    objective_fn = theano.function([input_image], -objective.define_loss(input_image, options.objective_model).mean())

    return generate_fn, objective_fn


def main():
    options = parse_args()
    import theano.sandbox.cuda
    theano.sandbox.cuda.use(options.device)

    plt.rcParams['image.cmap'] = 'gray'
    if not os.path.exists(options.output_folder):
        os.mkdir(options.output_folder)
    sys.path.insert(0, options.experiment)
    generate_fn, objective_fn = compile(options)
    import util
    X, ids = util.load_dataset(options.input_folder, False)
    mem_file = open(os.path.join(options.output_folder, 'mem.csv'), 'w')
    print ("id,from,to", file = mem_file)
    for id, img in zip(ids, X):
        input_img = util.preprocess(np.expand_dims(img, axis=0))
        output_img = generate_fn(util.add_noise(input_img))
        plt.imsave(os.path.join(options.output_folder, id), np.squeeze(util.deprocess(output_img)))
        line = "%s,%s,%s" % (id, np.squeeze(objective_fn(input_img)), np.squeeze(objective_fn(output_img)))
        print (line)
        print (line, file = mem_file)
    mem_file.close()

if __name__ == "__main__":
    main()
