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

    parser.add_argument("--experiment", default='memnet',
                        help="Experiment name, the folder with all network definitions")
    parser.add_argument("--num_iter", type=int, default=100, help="Number of iterations")
    parser.add_argument("--batch_size", type=int, default=4, help='Size of minibatch')
    parser.add_argument("--obj_coef", default=1.0, type=float, help="Weight of objective loss")
    parser.add_argument("--cont_coef", default=1.0, type=float, help="Weight of content loss")
    parser.add_argument("--disc_coef", default=10.0,  type=float, help="Weight of discriminator")
    parser.add_argument("--num_img_to_show", default=5, help="Number of image to show")
    parser.add_argument("--experiment_folder", default='memnet/experiment',
                        help='Folder for saving plots and netowkrs in each iteration')
    parser.add_argument("--log_file", default='memnet/plots-6/log.txt', help='Log file')
    parser.add_argument("--save_mode_it", default=1, help='Save model per this number of epoch')
    parser.add_argument("--device", default='gpu0', help='Which device to use')

    return parser.parse_args()


def compile(options):
    import content
    import discriminator
    import generator
    import objective

    input_to_generator = T.tensor4('img_with_noise', dtype='float32')
    input_to_content = T.tensor4('input_img', dtype='float32')
    input_to_disctiminator = T.tensor4('true_img', dtype='float32')
    lr = theano.shared(np.array(0.001, dtype='float32'))

    G = generator.define_net()
    generated_img = lasagne.layers.get_output(G['out'], inputs=input_to_generator)


    D = discriminator.define_net()
    D_true = lasagne.layers.get_output(D['out'], inputs = input_to_disctiminator)
    D_generated = lasagne.layers.get_output(D['out'], inputs = generated_img)

    objective_loss = options.obj_coef * objective.define_loss(generated_img).mean()
    content_loss = options.cont_coef * content.define_loss(input_to_content, generated_img).mean()
    disc_loss = -options.disc_coef * T.log(D_generated).mean()

    G_loss = (objective_loss +
              content_loss +
              disc_loss)

    true_loss = -T.log(D_true).mean()
    generated_loss = -T.log(1 - D_generated).mean()

    D_loss = true_loss + generated_loss

    D_params = lasagne.layers.get_all_params(D['out'], trainable=True)
    D_updates = lasagne.updates.adam(D_loss, D_params, learning_rate=lr)
    D_train_fn = theano.function([input_to_generator, input_to_disctiminator], [D_loss, true_loss, generated_loss],
                                 updates=D_updates, allow_input_downcast=True)

    G_params = lasagne.layers.get_all_params(G['out'], trainable=True)
    G_updates = lasagne.updates.adam(G_loss, G_params, learning_rate=lr)
    G_train_fn = theano.function([input_to_generator, input_to_content],
                                 [G_loss, objective_loss, content_loss, disc_loss], updates=G_updates,
                                 allow_input_downcast=True)
    generate_fn = theano.function([input_to_generator], generated_img, allow_input_downcast=True)
    return G_train_fn, D_train_fn, generate_fn, G, D, lr


def plot(options, epoch, images, generated_images):
    plt.clf()
    import util
    images = util.deprocess(images)
    generated_images = util.deprocess(generated_images)
    for i in range(len(images)):
        plt.subplot(len(images), 2, 2*i + 1)
        plt.axis('off')
        plt.imshow(images[i])
        plt.subplot(len(images), 2, 2*i + 2)
        plt.axis('off')
        plt.imshow(generated_images[i])
    plt.savefig(os.path.join(options.experiment_folder, 'plots', str(epoch) + '.png'), bbox_inches='tight')


def save_model(options, epoch, G, D):
    np.save(os.path.join(options.experiment_folder, 'model', 'generator-' + str(epoch) + '.npy'),
                lasagne.layers.get_all_param_values(G['out']))
    np.save(os.path.join(options.experiment_folder, 'model', 'discriminator-' + str(epoch) + '.npy'),
                lasagne.layers.get_all_param_values(D['out']))


def train(options):
    print ("Compiling...")
    G_train_fn, D_train_fn, generate_fn, G, D, lr = compile(options)
    import util
    print ("Loading dataset...")
    X = util.load_dataset()
    print ("Training...")
    log_file = open(options.log_file, 'w')
    for epoch in range(options.num_iter):
        if (epoch + 1) % 10 == 0:
            lr *= 0.25

        discriminator_order = np.arange(len(X))
        generator_order = np.arange(len(X))
        np.random.shuffle(discriminator_order)
        np.random.shuffle(generator_order)

        discriminator_loss_list = []
        generator_loss_list = []

        for start in range(0, len(X), options.batch_size):
            end = min(start + options.batch_size, len(X))

            generator_batch = util.preprocess(X[generator_order[start:end]])
            discriminator_batch = util.preprocess(X[discriminator_order[start:end]])

            generator_batch_with_noise = util.add_noise(generator_batch)

            #print (generate_fn(generator_batch_with_noise))
            generator_loss = G_train_fn(generator_batch_with_noise, generator_batch)
            discriminator_loss = D_train_fn(generator_batch_with_noise, discriminator_batch)

            discriminator_loss_list.append(discriminator_loss)
            generator_loss_list.append(generator_loss)

        img_for_ploting = util.preprocess(X[0:options.num_img_to_show])
        plot(options, epoch, img_for_ploting, generate_fn(util.add_noise(img_for_ploting)))

        log_str = (("Epoch %i" % epoch) + '\n'
                    + ("Discriminator loss %f, true_loss %f, generated_loss %f" %
                            tuple(np.mean(np.array(discriminator_loss_list), axis=0))) + '\n'
                    + ("Generator loss %f, obj_loss %f, cont_loss %f, disc_loss %f" %
                            tuple(np.mean(np.array(generator_loss_list), axis=0)))
                   )
        print(log_str)
        print(log_str, file=log_file)
        if epoch % options.save_mode_it == 0:
            save_model(options, epoch, G, D)
        log_file.flush()

    log_file.close()
    save_model(options, options.num_iter - 1, G, D)

    return G


def main():
    options = parse_args()
    import theano.sandbox.cuda
    theano.sandbox.cuda.use(options.device)
    plt.rcParams['image.cmap'] = 'gray'
    if not os.path.exists(options.experiment_folder):
        os.mkdir(options.experiment_folder)
    if not os.path.exists(os.path.join(options.experiment_folder, 'model')):
        os.mkdir(os.path.join(options.experiment_folder, 'model'))
    if not os.path.exists(os.path.join(options.experiment_folder, 'plots')):
        os.mkdir(os.path.join(options.experiment_folder, 'plots'))
    sys.path.insert(0, options.experiment)
    _ = train(options)

if __name__ == "__main__":
    main()
