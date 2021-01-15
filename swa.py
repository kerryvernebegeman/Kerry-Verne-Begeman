"""
Stochastic Weight Averaging: https://arxiv.org/abs/1803.05407
See: https://github.com/kristpapadopoulos/keras-stochastic-weight-averaging

Original: pbaylies
Modified: aydao

"""
import os
import glob
import pickle
import argparse

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from dnnlib.tflib import init_tf

filepath = 'output.pkl'

def fetch_models_from_files(model_list):
    for fn in model_list:
        print(fn)
        with open(fn, 'rb') as f:
            yield (fn, pickle.load(f))


def apply_swa_to_checkpoints(models):
    epoch = 0
    mod_gen = None
    mod_dis = None
    mod_gs = None
    for model_pair in models:
        fn, triple = model_pair
        print('Loading',fn, flush=True)
        gen, dis, gs = triple
        if mod_gen == None:
            mod_gen = gen
            mod_dis = dis
            mod_gs = gs
        else:
            mod_gen.apply_swa(gen, epoch)
            mod_dis.apply_swa(dis, epoch)
            mod_gs.apply_swa(gs, epoch)
        print("...next", flush=True)
        epoch += 1
    return (mod_gen, mod_dis, mod_gs)


parser = argparse.ArgumentParser(description='Perform stochastic weight averaging', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('results_dir', help='Directory with network checkpoints for weight averaging')
parser.add_argument('--filespec', default='*.pkl', help='The files to average')
parser.add_argument('--output_model', default='network_avg.pkl', help='The averaged model to output')

args, other_args = parser.parse_known_args()
filepath = args.output_model
files = glob.glob(os.path.join(args.results_dir,args.filespec))
files.sort()
swa_epochs = len(files)
print(swa_epochs, files)
init_tf()
models = fetch_models_from_files(files)
swa_models = apply_swa_to_checkpoints(models)

print('Final model parameters set to stochastic weight average.')
with open(filepath, 'wb') as f:
    pickle.dump(swa_models, f)
print('Final stochastic averaged weights saved to file.')
