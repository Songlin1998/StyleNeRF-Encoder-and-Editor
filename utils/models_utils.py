import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
import copy
import pickle
from argparse import Namespace
import os
import glob
import torch
import dnnlib
from configs import paths_config, global_config
from models.e4e.psp import pSp
from training.networks import Generator
from utils import legacy


def save_tuned_G(generator, w_pivots,cm_pivots, quads, run_id):
    generator = copy.deepcopy(generator).cpu()
    # pivots = copy.deepcopy(pivots).cpu()
    w_pivots =  copy.deepcopy(w_pivots).cpu()
    cm_pivots = copy.deepcopy(cm_pivots).cpu()
    torch.save({'generator': generator, 'w_pivots': w_pivots,'cm_pivots': cm_pivots , 'quads': quads},
               f'{paths_config.checkpoints_dir}/model_{run_id}.pt')


def load_tuned_G(run_id):
    new_G_path = f'{paths_config.checkpoints_dir}/model_{run_id}.pt'
    with open(new_G_path, 'rb') as f:
        checkpoint = torch.load(f)

    new_G, w_pivots,cm_pivots, quads = checkpoint['generator'], checkpoint['w_pivots'], checkpoint['cm_pivots'], checkpoint['quads']
    new_G = new_G.float().to(global_config.device).eval().requires_grad_(False)
    w_pivots = w_pivots.float().to(global_config.device)
    cm_pivots = cm_pivots.float().to(global_config.device)

    return new_G, w_pivots, cm_pivots, quads

def load_old_G_stylenerf():
    network_pkl = paths_config.stylegan2_ada_ffhq
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(global_config.device) # type: ignore
    G = copy.deepcopy(G).eval().requires_grad_(False).to(global_config.device) # type: ignore
    return G


def load_g(file_path):
    with open(file_path, 'rb') as f:
        old_G = pickle.load(f)['G_ema'].to(global_config.device).eval()
        old_G = old_G.float()
    return old_G


def initialize_e4e_wplus():
    ckpt = torch.load(paths_config.e4e, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = paths_config.e4e
    opts = Namespace(**opts)
    e4e_inversion_net = pSp(opts)
    e4e_inversion_net = e4e_inversion_net.eval().to(global_config.device).requires_grad_(False)
    return e4e_inversion_net

def initialize_naive_inverison():
    network_pkl = '/hd4/yangsonglin-3D/StyleNeRF/encoder_ckpt/checkpoints/network-snapshot-000280.pkl'
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        net = legacy.load_network_pkl(fp)['E']
    net = copy.deepcopy(net).eval().to(global_config.device).requires_grad_(False)
    return net


def initialize_deca_mix_e4e_adv_vgg_inversion():
    network_pkl = '/hd4/yangsonglin-3D/StyleNeRF/encoder_deca_mix_e4e_adv_vgg_ckpt_fine-tuning_4/checkpoints/000050.pkl'
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        E = legacy.load_network_pkl(fp)
    E_geo = copy.deepcopy(E['E_geo']).eval().to(global_config.device).requires_grad_(False)
    E_app = copy.deepcopy(E['E_app']).eval().to(global_config.device).requires_grad_(False)
    camera_mlp = copy.deepcopy(E['camera_mlp']).eval().to(global_config.device).requires_grad_(False)
    geometry_mlp = copy.deepcopy(E['geometry_mlp']).eval().to(global_config.device).requires_grad_(False)
    appearance_mlp = copy.deepcopy(E['appearance_mlp']).eval().to(global_config.device).requires_grad_(False)
    return E_geo,E_app,camera_mlp,geometry_mlp,appearance_mlp

def load_from_pkl_model(tuned):
    model_state = {'init_args': tuned.init_args, 'init_kwargs': tuned.init_kwargs
        , 'state_dict': tuned.state_dict()}
    gen = Generator(*model_state['init_args'], **model_state['init_kwargs'])
    gen.load_state_dict(model_state['state_dict'])
    gen = gen.eval().cuda().requires_grad_(False)
    return gen


def load_generators(run_id):
    tuned, w_pivots, cm_pivots, quads = load_tuned_G(run_id=run_id)
    original = load_old_G_stylenerf()
    gen = load_from_pkl_model(tuned)
    orig_gen = load_from_pkl_model(original)
    del tuned, original
    return gen, orig_gen, w_pivots,cm_pivots, quads
