#   ~~~ aydao ~~~~ 2020 ~~~
#
#   Create a .pkl for an uninitialized StyleGAN2 network
#

import argparse
import copy
import os
import sys

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import dnnlib
from dnnlib import EasyDict

from metrics.metric_defaults import metric_defaults

#----------------------------------------------------------------------------

def run(config_id, gamma, height, width, cond):
    train     = EasyDict(run_func_name='training.diagnostic.create_initial_pkl') # Options for training loop.
    G         = EasyDict(func_name='training.networks_stylegan2.G_main')       # Options for generator network.
    D         = EasyDict(func_name='training.networks_stylegan2.D_stylegan2')  # Options for discriminator network.
    G_opt     = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for generator optimizer.
    D_opt     = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for discriminator optimizer.
    G_loss    = EasyDict(func_name='training.loss.G_logistic_ns_pathreg')      # Options for generator loss.
    D_loss    = EasyDict(func_name='training.loss.D_logistic_r1')              # Options for discriminator loss.
    sched     = EasyDict()                                                     # Options for TrainingSchedule.
    grid      = EasyDict(size='8k', layout='random')                           # Options for setup_snapshot_image_grid().
    sc        = dnnlib.SubmitConfig()                                          # Options for dnnlib.submit_run().
    tf_config = {'rnd.np_random_seed': 1000}                                   # Options for tflib.init_tf().

    sched.minibatch_size_base = 192
    sched.minibatch_gpu_base = 3
    D_loss.gamma = 10
    desc = 'stylegan2'

    dataset_args = EasyDict() # (tfrecord_dir=dataset)

    if cond:
        desc += '-cond'; dataset_args.max_label_size = 'full' # conditioned on full label

    desc += '-' + config_id

    # Configs A-E: Shrink networks to match original StyleGAN.
    if config_id != 'config-f':
        G.fmap_base = D.fmap_base = 8 << 10

    # Config E: Set gamma to 100 and override G & D architecture.
    if config_id.startswith('config-e'):
        D_loss.gamma = 100
        if 'Gorig'   in config_id: G.architecture = 'orig'
        if 'Gskip'   in config_id: G.architecture = 'skip' # (default)
        if 'Gresnet' in config_id: G.architecture = 'resnet'
        if 'Dorig'   in config_id: D.architecture = 'orig'
        if 'Dskip'   in config_id: D.architecture = 'skip'
        if 'Dresnet' in config_id: D.architecture = 'resnet' # (default)

    # Configs A-D: Enable progressive growing and switch to networks that support it.
    if config_id in ['config-a', 'config-b', 'config-c', 'config-d']:
        sched.lod_initial_resolution = 8
        sched.G_lrate_base = sched.D_lrate_base = 0.001
        sched.G_lrate_dict = sched.D_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        sched.minibatch_size_base = 32 # (default)
        sched.minibatch_size_dict = {8: 256, 16: 128, 32: 64, 64: 32}
        sched.minibatch_gpu_base = 4 # (default)
        sched.minibatch_gpu_dict = {8: 32, 16: 16, 32: 8, 64: 4}
        G.synthesis_func = 'G_synthesis_stylegan_revised'
        D.func_name = 'training.networks_stylegan2.D_stylegan'

    # Configs A-C: Disable path length regularization.
    if config_id in ['config-a', 'config-b', 'config-c']:
        G_loss = EasyDict(func_name='training.loss.G_logistic_ns')

    # Configs A-B: Disable lazy regularization.
    if config_id in ['config-a', 'config-b']:
        train.lazy_regularization = False

    # Config A: Switch to original StyleGAN networks.
    if config_id == 'config-a':
        G = EasyDict(func_name='training.networks_stylegan.G_style')
        D = EasyDict(func_name='training.networks_stylegan.D_basic')

    if gamma is not None:
        D_loss.gamma = gamma

    G.update(resolution_h=height)
    G.update(resolution_w=width)
    D.update(resolution_h=height)
    D.update(resolution_w=width)

    sc.submit_target = dnnlib.SubmitTarget.DIAGNOSTIC
    sc.local.do_not_copy_source_files = True
    kwargs = EasyDict(train)
    kwargs.update(G_args=G, D_args=D, tf_config=tf_config, config_id=config_id,
        resolution_h=height, resolution_w=width)
    kwargs.submit_config = copy.deepcopy(sc)
    kwargs.submit_config.run_desc = desc
    # dnnlib.submit_run(**kwargs)
    dnnlib.submit_diagnostic(**kwargs)

#----------------------------------------------------------------------------

def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _parse_comma_sep(s):
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Creates a pkl of an uninitialized StyleGAN2 model.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', help='Training config (default: %(default)s)', default='config-f', required=False, dest='config_id', metavar='CONFIG')
    parser.add_argument('--gamma', help='R1 regularization weight (default is config dependent)', default=None, type=float)
    parser.add_argument('--height', help='dimension of height of input/output images (e.g., 1024)', default=1024, type=int)
    parser.add_argument('--width', help='dimension of width of input/output images (e.g., 1024)', default=1024, type=int)
    parser.add_argument('--cond', help='conditional model', default=False, metavar='BOOL', type=_str_to_bool)
    
    args = parser.parse_args()

    run(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------


