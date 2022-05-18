# From https://github.com/rosinality/stylegan2-pytorch/blob/master/ppl.py
import argparse

import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

import lpips
from utils import load_hparams, load_checkpoint


def normalize(x):
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True))


def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = (a * b).sum(-1, keepdim=True)
    p = t * torch.acos(d)
    c = normalize(b - d * a)
    d = a * torch.cos(p) + c * torch.sin(p)

    return normalize(d)


def lerp(a, b, t):
    return a + (b - a) * t


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perceptual Path Length calculator")

    parser.add_argument(
        "--space", choices=["z", "w"], default="w", help="space that PPL calculated with"
    )
    parser.add_argument(
        "--batch", type=int, default=64, help="batch size for the models"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=5000,
        help="number of the samples for calculating PPL",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--eps", type=float, default=1e-4, help="epsilon for numerical stability"
    )
    parser.add_argument(
        "--crop", action="store_true", help="apply center crop to the images"
    )
    parser.add_argument(
        "--sampling",
        default="end",
        choices=["end", "full"],
        help="set endpoint sampling method",
    )
    parser.add_argument('checkpoint_path', type=str, help='model to use')
    parser.add_argument('--device', type=str, default='cuda', help='model to use')
    parser.add_argument('--class_1', default=0, type=int, help='First class for interpolation')
    parser.add_argument('--class_2', default=1, type=int, help='Second class for interpolation')

    args = parser.parse_args()
    args.from_self_align = False
    device = args.device

    g, c_net = load_checkpoint(args)
    hparams = load_hparams(args.checkpoint_path)
    g.eval()
    c_net.eval()

    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )

    distances = []

    n_batch = args.n_sample // args.batch
    resid = args.n_sample - (n_batch * args.batch)
    batch_sizes = [args.batch] * n_batch + [resid]

    with torch.no_grad():
        for batch in tqdm(batch_sizes):

            z = torch.randn([batch * 2, hparams.code_size], device=device)
            if args.sampling == "full":
                lerp_t = torch.rand(batch, device=device)
            else:
                lerp_t = torch.zeros(batch, device=device)

            if args.space == "w":
                latent_t0, latent_t1 = c_net(torch.tensor(batch * [args.class_1], device=device)), c_net(torch.tensor(batch * [args.class_2], device=device))
                latent_e0 = lerp(latent_t0, latent_t1, lerp_t[:, None])
                latent_e1 = lerp(latent_t0, latent_t1, lerp_t[:, None] + args.eps)
                latent_e = torch.stack([latent_e0, latent_e1], 1).view(*[batch * 2, hparams.task_size])

            # image, _ = g([latent_e])
            image = g(z, step=6, alpha=1, task=latent_e)

            if args.crop:
                c = image.shape[2] // 8
                image = image[:, :, c * 3 : c * 7, c * 2 : c * 6]

            factor = image.shape[2] // 256

            if factor > 1:
                image = F.interpolate(
                    image, size=(256, 256), mode="bilinear", align_corners=False
                )

            dist = percept(image[::2], image[1::2]).view(image.shape[0] // 2) / (
                args.eps ** 2
            )
            distances.append(dist.to("cpu").numpy())

    distances = np.concatenate(distances, 0)

    lo = np.percentile(distances, 1, interpolation="lower")
    hi = np.percentile(distances, 99, interpolation="higher")
    filtered_dist = np.extract(
        np.logical_and(lo <= distances, distances <= hi), distances
    )

    print("ppl:", filtered_dist.mean())
