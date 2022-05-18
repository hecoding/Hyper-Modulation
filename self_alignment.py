import argparse
import random
import torch
from pathlib import Path
from tqdm import trange
from torch import nn, optim
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from model.stylegan import StyledGenerator as OriginalStyledGenerator
from utils import save_hparams, get_run_id


def get_z(batch_size, code_size, mixing, device, fixed_seed=False):
    rng = None
    if fixed_seed:
        rng = torch.Generator(device=device)
        rng.manual_seed(2147483647)
    if mixing and random.random() < 0.9:
        return [torch.randn(batch_size, code_size, generator=rng, device=device),
                torch.randn(batch_size, code_size, generator=rng, device=device)]
    else:
        return torch.randn(batch_size, code_size, generator=rng, device=device)


def get_noise(step, batch_size, device, fixed_seed=False):
    rng = None
    if fixed_seed:
        rng = torch.Generator(device=device)
        rng.manual_seed(2147483647)
    noise = []
    for i in range(step + 1):
        size = 4 * 2 ** i
        noise.append(torch.randn(batch_size, 1, size, size, generator=rng, device=device))

    return noise


def log_images(writer, i, generator, task, generator_original, code_size, step, device, batch_size=9):
    with torch.no_grad():
        z = get_z(batch_size=batch_size, code_size=code_size, mixing=False, device=device, fixed_seed=True)
        noise = get_noise(step, batch_size, device, fixed_seed=True)

        images_orig = generator_original(z, noise=noise, step=step, alpha=1, return_hierarchical=False)
        writer.add_image('sample_orig', make_grid(images_orig, nrow=3, normalize=True, range=(-1, 1)), i)

        task_in = task(torch.zeros(batch_size, device=device, dtype=torch.long))
        images = generator(z, noise=noise, step=step, alpha=1, task=task_in, return_hierarchical=False)
        writer.add_image('sample_new', make_grid(images, nrow=3, normalize=True, range=(-1, 1)), i)


def train(generator, task, generator_original, g_optimizer, criterion, iterations, args,
          batch_size=1, code_size=512, step=6, mixing=False, name=None, device='cpu', checkpoint_interval=5_000):
    name = 'unnamed' if name is None else name
    run_dir = f'{args.run_dir}/{get_run_id(args.run_dir):05d}-{name}'
    Path(run_dir).mkdir(parents=True)
    save_hparams(run_dir, args)
    writer = SummaryWriter(run_dir)
    alpha = 1
    for i in trange(iterations):
        g_optimizer.zero_grad()
        z = get_z(batch_size, code_size, mixing, device)
        noise = get_noise(step, batch_size, device)
        with torch.no_grad():
            _, g_l_orig = generator_original(z, noise=noise, step=step, alpha=alpha, return_hierarchical=True)
        task_in = task(torch.zeros(batch_size, device=device, dtype=torch.long))
        _, g_l = generator(z, noise=noise, step=step, alpha=alpha, task=task_in, return_hierarchical=True)

        assert len(g_l) == len(g_l_orig)
        g_loss = 0
        for input, target in zip(g_l, g_l_orig):
            g_loss += criterion(input, target)

        g_loss.backward()
        g_optimizer.step()

        writer.add_scalar('loss/g', g_loss, i)
        if i % args.log_interval == 0:
            log_images(writer, i, generator, task, generator_original, code_size, step, device)

        if i % checkpoint_interval == 0:
            torch.save({
                'generator': generator.state_dict(),
                'task': task.state_dict(),
            }, f'{run_dir}/{i:06d}.model')


def get_args():
    parser = argparse.ArgumentParser(description='Self-alignment')

    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--step', default=6, type=int, help='6 = 256px')
    parser.add_argument('--iterations', type=int, default=50_000, help='number of samples used')
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--code_size', default=512, type=int)
    parser.add_argument('--n_mlp_style', default=8, type=int)
    parser.add_argument('--n_mlp_task', default=4, type=int)
    parser.add_argument('--task_size', default=64, type=int)
    parser.add_argument('--batch_size', default=32, type=int, help='max image size')
    parser.add_argument('--mixing', action='store_true', help='use mixing regularization')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--run_dir', type=str, default='data/training-runs/self_align')
    parser.add_argument('--checkpoint_interval', type=int, default=5_000, help='number of samples used')
    parser.add_argument('--log_interval', type=int, default=500)

    return parser.parse_args()


def main(args):
    from model.hyper_mod import StyledGenerator, Task
    from pretrained_converter import convert_generator, assert_loaded_keys, freeze_layers

    args.origin = 'self_align'
    g_running_orig = OriginalStyledGenerator(args.code_size).to(args.device)
    g_running_orig.train(False)
    generator = StyledGenerator(code_dim=args.code_size, task_dim=args.task_size, n_mlp=args.n_mlp_style).to(args.device)
    freeze_layers(generator)
    task = Task(args.task_size, n_mlp=args.n_mlp_task, num_labels=1).to(args.device)

    g_optimizer = optim.Adam(generator.generator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    g_optimizer.add_param_group(
        {
            'params': generator.style.parameters(),
            'lr': args.lr * 0.01,
            'mult': 0.01,
        }
    )
    g_optimizer.add_param_group(
        {
            'params': task.parameters(),
            'lr': args.lr * 0.01,
            'mult': 0.001,
        }
    )

    ckpt = torch.load(args.checkpoint)
    g_running_orig.load_state_dict(ckpt['g_running'])
    missing, unexpected = generator.load_state_dict(convert_generator(ckpt['g_running']), strict=False)
    assert_loaded_keys(missing, unexpected)
    del ckpt

    criterion = nn.L1Loss()

    train(generator=generator, task=task, generator_original=g_running_orig, g_optimizer=g_optimizer,
          criterion=criterion, iterations=args.iterations, args=args, batch_size=args.batch_size,
          code_size=args.code_size, step=args.step, mixing=args.mixing, name=args.name, device=args.device,
          checkpoint_interval=args.checkpoint_interval)


if __name__ == '__main__':
    main(get_args())
