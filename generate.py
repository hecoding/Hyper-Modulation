import argparse
import math

import torch
from torchvision import utils

from utils import load_checkpoint


@torch.no_grad()
def get_mean_style(generator, device):
    mean_style = None

    for i in range(10):
        style = generator.mean_style(torch.randn(1024, 512).to(device))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10
    return mean_style

@torch.no_grad()
def sample(generator, class_network, n_class, step, mean_style, n_sample, device, seed):
    rng = torch.Generator()
    rng.manual_seed(seed)
    class_ = class_network(torch.tensor([n_class] * n_sample, device=device))
    image = generator(
        torch.randn(n_sample, 512, generator=rng).to(device),
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,
        task=class_,
    )
    
    return image

@torch.no_grad()
def style_mixing(generator, class_network, n_class, step, mean_style, n_source, n_target, device, seed):
    rng = torch.Generator()
    rng.manual_seed(seed)
    source_code = torch.randn(n_source, 512, generator=rng).to(device)
    target_code = torch.randn(n_target, 512, generator=rng).to(device)
    class_source = class_network(torch.tensor([n_class] * n_source, device=device))
    class_target = class_network(torch.tensor([n_class] * n_target, device=device))

    shape = 4 * 2 ** step
    alpha = 1

    images = [torch.ones(1, 3, shape, shape).to(device) * -1]

    source_image = generator(
        source_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7, task=class_source,
    )
    target_image = generator(
        target_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7, task=class_target,
    )

    images.append(source_image)

    for i in range(n_target):
        image = generator(
            [target_code[i].unsqueeze(0).repeat(n_source, 1), source_code],
            step=step,
            alpha=alpha,
            mean_style=mean_style,
            style_weight=0.7,
            mixing_range=(0, 1),
            task=class_source,
        )
        images.append(target_image[i].unsqueeze(0))
        images.append(image)

    images = torch.cat(images, 0)
    
    return images


def main(args):
    print('Loading model...', end=' ')
    generator, class_network = load_checkpoint(args)
    generator.eval()
    print('Loaded')

    mean_style = get_mean_style(generator, args.device)

    step = int(math.log(args.size, 2)) - 2

    print('Generating sample')
    img = sample(generator, class_network, args.class_num, step, mean_style, args.n_row * args.n_col, args.device, args.seed)
    utils.save_image(img, args.out, nrow=args.n_col, normalize=True, range=(-1, 1))

    if not args.no_mixing:
        print('Generating style mixing')
        for j in range(20):
            img = style_mixing(generator, class_network, args.class_num, step, mean_style, args.n_col, args.n_row, args.device, args.seed)
            utils.save_image(
                img, f'{args.out}_sample_mixing_{j}.png', nrow=args.n_col + 1, normalize=True, range=(-1, 1)
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2147483647, help='RNG seed')
    parser.add_argument('--size', type=int, default=1024, help='size of the image')
    parser.add_argument('--n_row', type=int, default=3, help='number of rows of sample matrix')
    parser.add_argument('--n_col', type=int, default=5, help='number of columns of sample matrix')
    parser.add_argument('checkpoint_path', type=str, help='path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda', help='')
    parser.add_argument('--no_mixing', action='store_true', help='Dont generate style mixing samples')
    parser.add_argument('--class_num', type=int, default=0, help='Which class to generate')
    parser.add_argument('--out', type=str, default='sample.png', help='')

    main(parser.parse_args())
