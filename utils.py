import argparse
import yaml
from pathlib import Path
import math
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pytorch_lightning import seed_everything as seed_everything_pl

HPARAMS_FILENAME = 'hparams.yml'


# ------------------------------------ I/O ------------------------------------


def args_to_yaml(path, args, exist_ok=False):
    file = Path(path)

    if file.exists():
        if exist_ok:
            return
        else:
            raise FileExistsError(f'File already exists {file}')

    with file.open('w') as f:
        yaml.dump(args.__dict__, f,
                  default_flow_style=False,
                  sort_keys=True)


def yaml_to_args(path):
    with open(path, 'r') as f:
        hparams = yaml.full_load(f)

    return argparse.Namespace(**hparams)


def save_hparams(path, args, exist_ok=False):
    hparams_file = Path(path) / HPARAMS_FILENAME
    args_to_yaml(hparams_file, args, exist_ok=exist_ok)


def load_hparams(path):
    path = Path(path)
    if not path.is_dir():
        path = path.parent

    hparams_file = path / HPARAMS_FILENAME

    if hparams_file.exists():
        return yaml_to_args(hparams_file)

    return None


def num_to_one_hot(t, num_classes=3):
    return torch.nn.functional.one_hot(torch.tensor(t) if isinstance(t, int) or isinstance(t, list) else t, num_classes=num_classes).float()


def load_checkpoint(args, hparams=None):
    if hparams is None:
        hparams = load_hparams(args.checkpoint_path)
    if hasattr(hparams, 'original_model') and hparams.original_model:
        from model.stylegan import StyledGenerator
        model = StyledGenerator(code_dim=hparams.code_size, n_mlp=hparams.n_mlp, c_dim=hparams.num_classes).to(args.device)
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint['g_running'] if 'g_running' in checkpoint else checkpoint)
        return model, num_to_one_hot
    from model.hyper_mod import StyledGenerator, Task

    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    model = StyledGenerator(code_dim=hparams.code_size, task_dim=hparams.task_size, n_mlp=hparams.n_mlp_style).to(args.device)
    task = Task(hparams.task_size, n_mlp=hparams.n_mlp_task, num_labels=hparams.num_classes).to(args.device)

    from_self_align = False
    if hasattr(hparams, 'origin') and hparams.origin == 'self_align':
        from_self_align = True

    if from_self_align:
        model.load_state_dict(checkpoint['generator'])
        task.load_state_dict(checkpoint['task'])
    else:
        model.load_state_dict(checkpoint['g_running'])
        task.load_state_dict(checkpoint['t_running_g'])
    return model, task


# ------------------------------------ Logging ------------------------------------


def get_run_id(outdir):
    import os, re
    # From StyleGAN repo
    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    return max(prev_run_ids, default=-1) + 1


def log_images(fname, alpha, args, dataset, resolution, batch_size, step, generator, task, device='cuda', seed=2147483647):
    if args.no_tb:
        return
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    images = []
    default_gen_n_classes = 8
    default_gen_n_samples = 4
    gen_n_classes, gen_n_samples = args.gen_sample.get(resolution, (default_gen_n_classes, default_gen_n_samples))
    if dataset.num_classes < gen_n_classes:
        # cycle through the classes
        gen_class_range = list(range(dataset.num_classes)) * math.ceil(gen_n_classes / dataset.num_classes)
        gen_class_range = gen_class_range[:gen_n_classes]
    else:
        gen_class_range = list(range(gen_n_classes))
    gen_class_range = gen_class_range * gen_n_samples
    with torch.no_grad():
        for j in range(0, len(gen_class_range), batch_size):
            gen_classes = gen_class_range[j:j + batch_size]
            images.append(
                generator(
                    torch.randn(len(gen_classes), args.code_size, generator=rng, device=rng.device),
                    step=step, alpha=alpha, task=task(torch.tensor(gen_classes).to(device))
                ).data.cpu()
            )

    save_image(torch.cat(images, 0), fname, nrow=gen_n_classes, normalize=True, range=(-1, 1))


# ------------------------------------ Training ------------------------------------


def seed_everything(seed):  # cleaner imports
    seed_everything_pl(seed)


def not_frozen_params(model):
    require = {}
    for name, param in model.named_parameters():
        require[name] = param.requires_grad

    return require


def requires_grad(model, flag=True, grad_map=None):
    for n, p in model.named_parameters():
        if flag and grad_map is not None and not grad_map[n]:  # filter those params which were originally frozen
            continue

        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999, grad_map=None):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        if grad_map is not None and not grad_map['module.' + k]:  # filter out frozen params
            continue

        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(dataset, batch_size, image_size=4):
    dataset.resolution = image_size
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1, drop_last=True)

    return loader


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult


# ------------------------------------ Metrics ------------------------------------


def load_scores(path):
    ms = {}
    for m_f in Path(path).glob('*.metric'):
        with open(m_f, 'rb') as f:
            data = pickle.load(f)
        ms[int(m_f.stem)] = data
    return dict(sorted(ms.items()))


def get_submetric(s):
    parts = s.rsplit('.', 1)
    if len(parts) == 1:
        return s, None
    if parts[1] == '':
        raise ValueError('dot is used to mark metric index, dont use at the end')
    return parts[0], parts[1]


def class_mean(ms, metric='fid', return_steps=False):
    ms = load_scores(ms)
    steps = list(ms.keys())
    class_m = 0
    classes = ms[steps[0]].keys()
    metric, sub_metric = get_submetric(metric)
    for class_i in classes:
        if sub_metric is not None:
            class_m += np.array([v[class_i][metric][sub_metric] for k, v in ms.items()])
        else:
            class_m += np.array([v[class_i][metric] for k, v in ms.items()])

    if return_steps:
        return steps, class_m / len(classes)
    return class_m / len(classes)
