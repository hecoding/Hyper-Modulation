import argparse
import torch
from torchvision.utils import save_image
from utils import seed_everything, load_hparams, load_checkpoint


def slerp(low, high, val):
    # https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
    low_norm = low / torch.norm(low, dim=-1, keepdim=True)
    high_norm = high / torch.norm(high, dim=-1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(-1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(-1) * low + (torch.sin(val * omega) / so).unsqueeze(-1) * high
    return res


def class_to_class(args, hparams, class_network, generator, device):
    images = []

    print(f'Generating interpolation with {args.steps} steps between class {args.class_1} and {args.class_2}, seed {args.seed}')
    with torch.no_grad():
        class1 = class_network(torch.tensor(args.class_1, device=device))
        class2 = class_network(torch.tensor(args.class_2, device=device))
        for interpolation_i in range(args.num_interpolations):
            z = torch.randn(1, hparams.code_size, device=device)
            for alpha in torch.linspace(0, 1, steps=args.steps, device=device):
                if args.interpolation == 'linear':
                    class_interpolation = class1.lerp(class2, alpha)
                else:
                    class_interpolation = slerp(class1, class2, alpha)
                image = generator(z, step=6, alpha=1, task=class_interpolation)
                images.append(image.cpu())

    return images


def z_to_z(args, hparams, class_network, generator, device):
    images = []

    print(f'Generating stuff')
    with torch.no_grad():
        class1 = class_network(torch.tensor(args.class_1, device=device))
        for interpolation_i in range(args.num_interpolations):
            z1 = torch.randn(1, hparams.code_size, device=device)
            z2 = torch.randn(1, hparams.code_size, device=device)
            for alpha in torch.linspace(0, 1, steps=args.steps, device=device):
                if args.interpolation == 'linear':
                    z_interpolation = z1.lerp(z2, alpha)
                else:
                    z_interpolation = slerp(z1, z2, alpha)
                image = generator(z_interpolation, step=6, alpha=1, task=class1)
                images.append(image.cpu())

    return images


def class_noise(args, hparams, class_network, generator, device):
    images = []

    print(f'Generating stuff, seed {args.seed}')
    with torch.no_grad():
        class1 = class_network(torch.tensor(args.class_1, device=device))
        class2 = class_network(torch.tensor(args.class_2, device=device))
        z1 = torch.randn(1, hparams.code_size, device=device)
        z2 = torch.randn(1, hparams.code_size, device=device)
        for alpha_z in torch.linspace(0, 1, steps=args.steps, device=device):
            if args.interpolation == 'linear':
                z_interpolation = z1.lerp(z2, alpha_z)
            else:
                z_interpolation = slerp(z1, z2, alpha_z)
            for alpha_class in torch.linspace(0, 1, steps=args.steps, device=device):
                if args.interpolation == 'linear':
                    class_interpolation = class1.lerp(class2, alpha_class)
                else:
                    class_interpolation = slerp(class1, class2, alpha_class)
                image = generator(z_interpolation, step=6, alpha=1, task=class_interpolation)
                images.append(image.cpu())

    return images


def random_z(args, hparams, class_network, generator, device):
    images = []

    print(f'Generating {args.steps * args.num_interpolations} from class {args.class_1} with random z, seed {args.seed}')
    with torch.no_grad():
        class1 = class_network(torch.tensor(args.class_1, device=device))
        for i in range(args.steps * args.num_interpolations):
            z = torch.randn(1, hparams.code_size, device=device)
            image = generator(z, step=6, alpha=1, task=class1)
            images.append(image.cpu())

    return images


def noise_shift(args, hparams, class_network, generator, device):
    images = []

    print(f'Generating stuff')
    with torch.no_grad():
        class1 = class_network(torch.tensor(args.class_1, device=device))
        for interpolation_i in range(args.num_interpolations):
            z = torch.randn(1, hparams.code_size, device=device)
            for alpha in torch.linspace(-1, 1, steps=args.steps, device=device):
                image = generator(z + alpha, step=6, alpha=1, task=class1)
                images.append(image.cpu())

    return images


def class_shift(args, hparams, class_network, generator, device):
    images = []

    print(f'Generating stuff')
    with torch.no_grad():
        class1 = class_network(torch.tensor(args.class_1, device=device))
        for interpolation_i in range(args.num_interpolations):
            z = torch.randn(1, hparams.code_size, device=device)
            for alpha in torch.linspace(-0.1, 0.1, steps=args.steps, device=device):
                image = generator(z, step=6, alpha=1, task=class1 + alpha)
                images.append(image.cpu())

    return images


def main(args):
    seed_everything(args.seed)
    device = args.device
    print(f'Loading model {args.checkpoint_path}')
    args.from_self_align = False
    generator, class_network = load_checkpoint(args)
    hparams = load_hparams(args.checkpoint_path)
    print('Loaded')
    class_network.eval()
    generator.eval()

    if args.type == 'class':
        images = class_to_class(args, hparams, class_network, generator, device)
    elif args.type == 'noise':
        images = z_to_z(args, hparams, class_network, generator, device)
    elif args.type == 'class_noise':
        images = class_noise(args, hparams, class_network, generator, device)
    elif args.type == 'random_z':
        images = random_z(args, hparams, class_network, generator, device)
    elif args.type == 'z_shift':
        images = noise_shift(args, hparams, class_network, generator, device)
    elif args.type == 'class_shift':
        images = class_shift(args, hparams, class_network, generator, device)
    else:
        raise ValueError

    save_image(torch.cat(images, 0), args.output, nrow=args.steps, padding=0, normalize=True, range=(-1, 1))
    print(f'Saved to {args.output}')


def get_args():
    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    parser.add_argument('checkpoint_path', type=str, help='model to use')
    parser.add_argument('--output', default='interpolation.png', type=str, help='Output file')
    parser.add_argument('--steps', default=10, type=int, help='Number of interpolations')
    parser.add_argument('--seed', default=2147483647, type=int, help='Random seed')
    parser.add_argument('--device', default='cuda', type=str, help='Device to use')
    parser.add_argument('--class_1', default=0, type=int, help='First class for interpolation')
    parser.add_argument('--class_2', default=1, type=int, help='Second class for interpolation')
    parser.add_argument('--interpolation', default='linear', type=str, choices=['linear', 'spherical'], help='Interpolation type')
    parser.add_argument('--num_interpolations', default=1, type=int, help='How many interpolations to perform')
    parser.add_argument('--type', default='class', type=str, help='Which interpolation to do')

    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())
