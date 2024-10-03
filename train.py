import os
os.environ['CUDA_VISIBLE_DEVICES']='8'
import json

from tqdm import tqdm, trange

from utils.models_utils import save_tuned_G

from datasets.image_list_dataset import ImageListDataset
from training.coaches.coach import Coach
from utils.data_utils import make_dataset

import click
import numpy as np
import torch
import wandb
from PIL import Image
from torchvision import transforms
import models
from configs import paths_config, global_config, hyperparameters
from utils.alignment import crop_faces, calc_alignment_coefficients
from utils.morphology import dilation
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor
import models.seg_model_2
import copy
from torch.utils.tensorboard import SummaryWriter
from module.flow import cnf
'''
python train.py --input_folder /hd4/yangsonglin-3D/STIT/datasets/obama_mini --output_folder /hd4/yangsonglin-3D/STIT/output_paper/obama_inversion_nerf_3da_0 --run_name obama --num_pti_steps 80
'''
def masked_l2(input, target, mask, loss_l2):
    loss = torch.nn.MSELoss if loss_l2 else torch.nn.L1Loss
    criterion = loss(reduction='none')
    masked_input = input * mask
    masked_target = target * mask
    error = criterion(masked_input, masked_target)
    return error.sum() / mask.sum()

def logical_or_reduce(*tensors):
    return torch.stack(tensors, dim=0).any(dim=0)

def logical_and_reduce(*tensors):
    return torch.stack(tensors, dim=0).all(dim=0)

def create_masks(border_pixels, mask, inner_dilation=0, outer_dilation=0, whole_image_border=False):
    image_size = mask.shape[2]
    grid = torch.cartesian_prod(torch.arange(image_size), torch.arange(image_size)).view(image_size, image_size,
                                                                                         2).cuda()
    image_border_mask = logical_or_reduce(
        grid[:, :, 0] < border_pixels,
        grid[:, :, 1] < border_pixels,
        grid[:, :, 0] >= image_size - border_pixels,
        grid[:, :, 1] >= image_size - border_pixels
    )[None, None].expand_as(mask)

    temp = mask
    if inner_dilation != 0:
        temp = dilation(temp, torch.ones(2 * inner_dilation + 1, 2 * inner_dilation + 1, device=mask.device),
                        engine='convolution')

    border_mask = torch.min(image_border_mask, temp)
    full_mask = dilation(temp, torch.ones(2 * outer_dilation + 1, 2 * outer_dilation + 1, device=mask.device),
                         engine='convolution')
    if whole_image_border:
        border_mask_2 = 1 - temp
    else:
        border_mask_2 = full_mask - temp
    border_mask = torch.maximum(border_mask, border_mask_2)

    border_mask = border_mask.clip(0, 1)
    content_mask = (mask - border_mask).clip(0, 1)
    return content_mask, border_mask, full_mask


def calc_masks(inversion, segmentation_model, border_pixels, inner_mask_dilation, outer_mask_dilation,
               whole_image_border):
    background_classes = [0, 18, 16]
    inversion_resized = torch.cat([F.interpolate(inversion, (512, 512), mode='nearest')])
    inversion_normalized = transforms.functional.normalize(inversion_resized.clip(-1, 1).add(1).div(2),
                                                           [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    segmentation = segmentation_model(inversion_normalized)[0].argmax(dim=1, keepdim=True)
    is_foreground = logical_and_reduce(*[segmentation != cls for cls in background_classes])
    foreground_mask = is_foreground.float()
    content_mask, border_mask, full_mask = create_masks(border_pixels // 2, foreground_mask, inner_mask_dilation // 2,
                                                        outer_mask_dilation // 2, whole_image_border)
    content_mask = F.interpolate(content_mask, (1024, 1024), mode='bilinear', align_corners=True)
    border_mask = F.interpolate(border_mask, (1024, 1024), mode='bilinear', align_corners=True)
    full_mask = F.interpolate(full_mask, (1024, 1024), mode='bilinear', align_corners=True)
    return content_mask, border_mask, full_mask

def save_image(image: Image.Image, output_folder, image_name, image_index, ext='jpg'):
    if ext == 'jpeg' or ext == 'jpg':
        image = image.convert('RGB')
    folder = os.path.join(output_folder, image_name)
    os.makedirs(folder, exist_ok=True)
    image.save(os.path.join(folder, f'{image_index}.{ext}'))


def paste_image(coeffs, img, orig_image):
    pasted_image = orig_image.copy().convert('RGBA')
    projected = img.convert('RGBA').transform(orig_image.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    pasted_image.paste(projected, (0, 0), mask=projected)
    return pasted_image


def to_pil_image(tensor: torch.Tensor) -> Image.Image:
    x = (tensor[0].permute(1, 2, 0) + 1) * 255 / 2
    x = x.detach().cpu().numpy()
    x = np.rint(x).clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)


@click.command()
@click.option('-i', '--input_folder', type=str, help='Path to (unaligned) images folder', required=True)
@click.option('-o', '--output_folder', type=str, help='Path to output folder', required=True)
@click.option('--start_frame', type=int, default=0)
@click.option('--end_frame', type=int, default=None)
@click.option('-r', '--run_name', type=str, required=True)
@click.option('--use_fa/--use_dlib', default=False, type=bool)
@click.option('--scale', default=1.0, type=float)
@click.option('--num_pti_steps', default=300, type=int)
@click.option('--l2_lambda', type=float, default=10.0)
@click.option('--center_sigma', type=float, default=1.0)
@click.option('--xy_sigma', type=float, default=3.0)
@click.option('--pti_learning_rate', type=float, default=3e-5)
@click.option('--use_locality_reg/--no_locality_reg', type=bool, default=False)
@click.option('--use_wandb/--no_wandb', type=bool, default=False)
@click.option('--pti_adam_beta1', type=float, default=0.9)
@click.option('--freeze_fine_layers', type=int, default=None)
@click.option('--outer_mask_dilation', type=int, default=50)
@click.option('--inner_mask_dilation', type=int, default=0)
@click.option('--whole_image_border', is_flag=True, type=bool)
def main(**config):
    _main(**config, config=config)


def _main(input_folder, output_folder, start_frame, end_frame, run_name,
          scale, num_pti_steps, l2_lambda, center_sigma, xy_sigma,
          use_fa, use_locality_reg, use_wandb, config, pti_learning_rate, pti_adam_beta1,
          freeze_fine_layers,outer_mask_dilation,inner_mask_dilation, whole_image_border):
    global_config.run_name = run_name
    hyperparameters.max_pti_steps = num_pti_steps
    hyperparameters.pt_l2_lambda = l2_lambda
    hyperparameters.use_locality_regularization = use_locality_reg
    hyperparameters.pti_learning_rate = pti_learning_rate
    hyperparameters.pti_adam_beta1 = pti_adam_beta1
    if use_wandb:
        wandb.init(project=paths_config.pti_results_keyword, reinit=True, name=global_config.run_name, config=config)
    files = make_dataset(input_folder)
    files = files[start_frame:end_frame]
    
    print(f'Number of images: {len(files)}')
    image_size = 1024
    print('Aligning images')
    crops, orig_images, quads = crop_faces(image_size, files, scale,
                                           center_sigma=center_sigma, xy_sigma=xy_sigma, use_fa=use_fa)
    print('Aligning completed')

    ds = ImageListDataset(crops, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    coach = Coach(ds, use_wandb)

    print('start pivot tuning')
    w_pivots, cm_pivots = coach.train()
    print('finish pivot tuning')
    
    save_tuned_G(coach.G, w_pivots, cm_pivots, quads, global_config.run_name)

    inverse_transforms = [
        calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]])
        for quad in quads]

    pit_gen = coach.G.requires_grad_(False).eval().to(global_config.device)
    ori_gen = coach.original_G.requires_grad_(False).eval().to(global_config.device)
    # stit_gen = copy.deepcopy(pit_gen).eval().requires_grad_(False).to(global_config.device)

    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, 'opts.json'), 'w') as f:
        json.dump(config, f)
   
    segmentation_model = models.seg_model_2.BiSeNet(19).eval().cuda().requires_grad_(False)
    segmentation_model.load_state_dict(torch.load(paths_config.segmentation_model_path))

    for i, (coeffs, crop, orig_image, w, cm) in tqdm(
            enumerate(zip(inverse_transforms, crops, orig_images, w_pivots, cm_pivots)), total=len(w_pivots)):
        w = w[None]
        cm = cm[None]
        with torch.no_grad():
            border_pixels = outer_mask_dilation
            crop_tensor = to_tensor(crop)[None].mul(2).sub(1).cuda()
            content_mask, border_mask, full_mask = calc_masks(crop_tensor, segmentation_model, border_pixels,
                                                                inner_mask_dilation, outer_mask_dilation,
                                                                whole_image_border)
            
            batch = w.shape[0]
            c_samples = cm.to(global_config.device)
            camera_matrices = pit_gen.synthesis.get_camera(batch, global_config.device, mode=c_samples)
            gen_img = pit_gen.get_final_output(styles=w, camera_matrices=camera_matrices)
            ori_img = ori_gen.get_final_output(styles=w, camera_matrices=camera_matrices)
            # background_mask_img = (border_mask + content_mask) * gen_img + (1-border_mask-content_mask)* crop_tensor
            background_mask_img = content_mask * gen_img + (1-content_mask)* crop_tensor
            
            gen_img = to_pil_image(gen_img)
            background_mask_img = to_pil_image(background_mask_img)
            ori_img = to_pil_image(ori_img)

        save_image(paste_image(coeffs, gen_img, orig_image), output_folder, 'gen_img', i)
        save_image(paste_image(coeffs, ori_img, orig_image), output_folder, 'ori_img', i)
        save_image(paste_image(coeffs, background_mask_img, orig_image), output_folder, 'background_mask_img', i)

if __name__ == '__main__':
    main()
