#
#   ~~ StyleGAN Surgery ~~
#         A Small Incision, and Other "Elective" Operations
#   ~~~ aydao ~~~~ 2020 ~~~
#
import os
import sys
import pickle
import argparse
import numpy as np
import PIL.Image
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import dnnlib
import dnnlib.tflib as tflib

def main(args):
    operation = args.operation
    input_pkl = args.input_pkl
    results_dir = args.results_dir
    output_pkl = args.output_pkl
    if output_pkl == None or output_pkl == 'None':
        output_pkl = input_pkl.replace('.pkl','-'+operation+'.pkl')

    tflib.init_tf()

    G, D, Gs = pickle.load(open(input_pkl, 'rb'))
    G.print_layers()
    D.print_layers() 
    Gs.print_layers()
    
    update_weights = []
    layers = [v for v in tf.trainable_variables() if 'const' in v.name]
    for x in layers:
        w = x
        if operation == 'upend':
            w_1 = tf.transpose(x, (0, 2, 3, 1))
            w_2 = tf.image.rot90(w_1, 2)
            w = tf.transpose(w_2, (0, 3, 1, 2))
        elif operation == 'unfold':
            w_1 = x[:,:,:,0:x.shape[3]//2]
            w_2 = x[:,:,:,x.shape[3]//2:]
            w = tf.concat([w_2, w_1], 3)
        elif operation == 'normalize':
            # w = tf.linalg.normalize(x)
            w = x/tf.reduce_max(tf.abs(x))
        elif operation == 'zeros':
            w = tf.zeros(x.shape)
        elif operation == 'halfzeros':
            w_1 = x[:,:,:,0:x.shape[3]//2]
            z_2 = tf.zeros((x.shape[0],x.shape[1],x.shape[2],x.shape[3]//2))
            w = tf.concat([w_1, z_2], 3)
        elif operation == 'threequarterzeros':
            w_1 = x[:,:,:,0:x.shape[3]//4]
            z_2 = tf.zeros((x.shape[0],x.shape[1],x.shape[2],3*(x.shape[3]//4)))
            w = tf.concat([w_1, z_2], 3)
            # mask = tf.less(x, 0.1 * tf.ones_like(x))
            # w = tf.multiply(x, tf.cast(mask, tf.float32))
        else:
            print('Unregistered operation',operation)
            sys.exit(-1)
        update_weight = [tf.assign(x, w)]
        update_weights.append(update_weight)
    
    tf.get_default_session().run(update_weights)
    
    filename = output_pkl
    with open(filename, 'wb') as file:
        pickle.dump((G, D, Gs), file, protocol=pickle.HIGHEST_PROTOCOL)
    
    name = filename.replace('.pkl','')
    
    if results_dir != None and results_dir != "None":
        for i in range(0,100):
            rnd = np.random.RandomState(None)
            latents = rnd.randn(1, Gs.input_shape[1])
            fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
            images = Gs.run(latents, None, truncation_psi=0.6, randomize_noise=True, output_transform=fmt)
            dir = results_dir + name + '/'
            os.makedirs(dir, exist_ok=True)
            png_filename = os.path.join(dir, 'example-'+str(i)+'.png')
            PIL.Image.fromarray(images[0], 'RGB').save(png_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform StyleGAN surgery', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_pkl', help='The StyleGAN2 input pkl')
    parser.add_argument('--operation', default='unfold', help='The type of surgical operation to perform')
    parser.add_argument('--results_dir', default='None', help='Directory in which to generate samples')
    parser.add_argument('--output_pkl', default='None', help='The modified model, post the StyleGAN surgical operation.')

    args = parser.parse_args()
    main(args)
