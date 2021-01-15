#   ~~~ aydao ~~~~ 2020 ~~~
#
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

# Note well that the argument order is target then source  
def copy_and_crop_trainables_from(target_net, source_net) -> None:
    source_trainables = source_net.trainables.keys()
    target_trainables = target_net.trainables.keys()
    names = [pair for pair in zip(source_trainables, target_trainables)]
            
    skip = []
    for pair in names:
        source_name, target_name = pair
        x = source_net.get_var(source_name)
        y = target_net.get_var(target_name)
        source_shape = x.shape
        target_shape = y.shape
        if source_shape != target_shape:
            update = x
            index = None 
            if 'Dense' in source_name:
                index = 0
                gap = source_shape[index] - target_shape[index]
                start = abs(gap) // 2
                end = start + target_shape[index]
                update = update[start:end,:]
            else:
                if source_shape[2] != target_shape[2]:
                    index = 2
                    gap = source_shape[index] - target_shape[index]
                    start = abs(gap) // 2
                    end = start + target_shape[index]
                    update = update[:,:,start:end,:]
                if source_shape[3] != target_shape[3]:
                    index = 3
                    gap = source_shape[index] - target_shape[index]
                    start = abs(gap) // 2
                    end = start + target_shape[index]
                    update = update[:,:,:,start:end]

            target_net.set_var(target_name, update)
            skip.append(source_name)

    weights_to_copy = {target_net.vars[pair[1]]: source_net.vars[pair[0]] for pair in names if pair[0] not in skip}
    tfutil.set_vars(tfutil.run(weights_to_copy))

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
            
            copy_and_crop_trainables_from(targetG, sourceG)
            copy_and_crop_trainables_from(targetD, sourceD)
            copy_and_crop_trainables_from(targetGs, sourceGs)
            
            misc.save_pkl((targetG, targetD, targetGs), os.path.join('./', output_pkl))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Copy and crop weights from one StyleGAN pkl to another', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('source_pkl', help='Path to the source pkl (weights copied from this one). This will *not* be overwritten or modified.')
    parser.add_argument('target_pkl', help='Path to the target pkl (weights copied onto this one). This will *not* be overwritten or modified.')
    parser.add_argument('--output_pkl', default='network-copyover.pkl', help='Path to the output pkl (source_pkl weights copied into target_pkl architecture)')
    args = parser.parse_args()
    main(args)