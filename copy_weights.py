#   ~~~ aydao ~~~~ 2020 ~~~
#
#   Copy over the weights (trainables in G, D, Gs) to another network
#
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import sys, getopt, os

import numpy as np
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib import tfutil
from dnnlib.tflib.autosummary import autosummary

from training import dataset
from training import misc
import pickle
import argparse

def main(args):

    source_pkl = args.source_pkl
    target_pkl = args.target_pkl
    output_pkl = args.output_pkl

    tflib.init_tf()

    with tf.Session() as sess:
        with tf.device('/gpu:0'):

            sourceG, sourceD, sourceGs = pickle.load(open(source_pkl, 'rb'))
            targetG, targetD, targetGs = pickle.load(open(target_pkl, 'rb'))
            
            print('Source:')
            sourceG.print_layers()
            sourceD.print_layers() 
            sourceGs.print_layers()
            
            print('Target:')
            targetG.print_layers()
            targetD.print_layers() 
            targetGs.print_layers()
            
            # Note: originally used copy_trainables_from()
            # so, that function may be more battle-tested ...  
            targetG.copy_compatible_trainables_from(sourceG)
            targetD.copy_compatible_trainables_from(sourceD)
            targetGs.copy_compatible_trainables_from(sourceGs)
            
            misc.save_pkl((targetG, targetD, targetGs), os.path.join('./', output_pkl))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Copy weights from one StyleGAN pkl to another', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('source_pkl', help='Path to the source pkl (weights copied from this one). This will *not* be overwritten or modified.')
    parser.add_argument('target_pkl', help='Path to the target pkl (weights copied onto this one). This will *not* be overwritten or modified.')
    parser.add_argument('--output_pkl', default='network-copyover.pkl', help='Path to the output pkl (source_pkl weights copied into target_pkl architecture)')
    args = parser.parse_args()
    main(args)