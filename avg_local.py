#   ~~~ aydao ~~~~ 2020 ~~~
#
#   Based on pbaylies' swa.py script
#   except that this computes the average instead of the moving average 
#   and does so locally in this script, rather than by modifying network.py
#
import os
import glob
import pickle
import argparse

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import dnnlib.tflib as tflib
from dnnlib.tflib import tfutil

def add_networks(dst_net, src_net):
    names = []
    for name in dst_net.trainables.keys():
        if name not in src_net.trainables:
            print('Not restoring (not present):     {}'.format(name))
        elif dst_net.trainables[name].shape != src_net.trainables[name].shape:
            print('Not restoring (different shape): {}'.format(name))

        if name in src_net.trainables and dst_net.trainables[name].shape == src_net.trainables[name].shape:
            names.append(name)

    tfutil.set_vars(tfutil.run({dst_net.vars[name]: dst_net.vars[name] + src_net.vars[name] for name in names}))
    return dst_net

def apply_denominator(dst_net, denominator):
    denominator_inv = 1.0 / denominator
    names = [name for name in dst_net.trainables.keys()]
    tfutil.set_vars(tfutil.run({dst_net.vars[name]: dst_net.vars[name] * denominator_inv for name in names}))
    return dst_net

def main(args):
    
    filepath = args.output_model
    files = glob.glob(os.path.join(args.results_dir,args.filespec))
    files.sort()
    network_count = len(files)
    print('Discovered %s networks' % str(network_count))
    
    tflib.init_tf()

    avg_G, avg_D, avg_Gs = None, None, None
    for pkl_file in files:
        G, D, Gs = pickle.load(open(pkl_file, 'rb'))
        if avg_G == None:
            avg_G, avg_D, avg_Gs = G, D, Gs
        else:
            avg_G = add_networks(avg_G, G)
            avg_D = add_networks(avg_D, D)
            avg_Gs = add_networks(avg_Gs, Gs)
    
    apply_denominator(avg_G, network_count)
    apply_denominator(avg_D, network_count)
    apply_denominator(avg_Gs, network_count)

    models = (avg_G, avg_D, avg_Gs)

    print('Final model parameters set to weight average.')
    with open(filepath, 'wb') as f:
        pickle.dump(models, f)
    print('Final averaged weights saved to file.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Perform weight averaging', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('results_dir', help='Directory with network checkpoints for weight averaging')
    parser.add_argument('--filespec', default='*.pkl', help='The files to average')
    parser.add_argument('--output_model', default='network-avg.pkl', help='The averaged model to output')

    args = parser.parse_args()

    main(args)
