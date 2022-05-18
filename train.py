import argparse
from pathlib import Path
import random
import math

from tqdm import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import grad
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from dataset import MultiResolutionDataset
from model.hyper_mod import StyledGenerator, Task, Discriminator
from pretrained_converter import convert_generator, assert_loaded_keys, freeze_layers
from barlow_twins import BarlowTwins, DiffTransform as ContrTransform, compute_contrastive
from utils import seed_everything, save_hparams, sample_data, adjust_lr, not_frozen_params, requires_grad, accumulate, log_images, get_run_id


def train(args, dataset, generator, discriminator, task, g_running, t_running, g_optimizer, d_optimizer,
          contr_transform, contr_criterion):
    step = int(math.log2(args.init_size)) - 2
    resolution = 4 * 2 ** step
    loader = sample_data(
        dataset, args.batch.get(resolution, args.batch_default), resolution
    )
    data_loader = iter(loader)

    adjust_lr(g_optimizer, args.lr.get(resolution, 0.002))
    adjust_lr(d_optimizer, args.lr.get(resolution, 0.002))

    pbar = tqdm(range(args.iterations))

    grad_map_generator = not_frozen_params(generator)
    grad_map_discriminator = not_frozen_params(discriminator)
    requires_grad(generator, False)
    requires_grad(discriminator, True, grad_map=grad_map_discriminator)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    used_sample = 0
    num_imgs = 0

    max_step = int(math.log2(args.max_size)) - 2
    final_progress = False

    run_dir = f'{args.run_dir}/{get_run_id(args.run_dir):05d}-{args.name}'
    Path(run_dir).mkdir(parents=True)
    save_hparams(run_dir, args)
    if not args.no_tb:
        writer = SummaryWriter(run_dir)

    for i in pbar:
        d_optimizer.zero_grad()

        if args.const_alpha is not None:
            alpha = args.const_alpha
        else:
            alpha = min(1, 1 / args.phase * (used_sample + 1))

        if (resolution == args.init_size and args.ckpt is None) or final_progress:
            if args.const_alpha is not None:
                alpha = args.const_alpha
            else:
                alpha = 1

        if used_sample > args.phase * 2:
            used_sample = 0
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True
                ckpt_step = step + 1

            else:
                if args.const_alpha is not None:
                    alpha = args.const_alpha
                else:
                    alpha = 0
                ckpt_step = step

            resolution = 4 * 2 ** step

            loader = sample_data(
                dataset, args.batch.get(resolution, args.batch_default), resolution
            )
            data_loader = iter(loader)

            if not args.no_checkpointing:
                torch.save(
                    {
                        'generator': generator.module.state_dict(),
                        'discriminator': discriminator.module.state_dict(),
                        'g_optimizer': g_optimizer.state_dict(),
                        'd_optimizer': d_optimizer.state_dict(),
                        'g_running': g_running.state_dict(),
                        'task_g': task.module.state_dict(),
                        't_running_g': t_running.state_dict(),
                    },
                    f'{run_dir}/train_step-{ckpt_step}.model',
                )

            adjust_lr(g_optimizer, args.lr.get(resolution, 0.002))
            adjust_lr(d_optimizer, args.lr.get(resolution, 0.002))

        try:
            real_image, real_label = next(data_loader)

        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image, real_label = next(data_loader)

        used_sample += real_image.shape[0]
        num_imgs += real_image.shape[0]

        b_size = real_image.size(0)
        real_image = real_image.cuda()
        real_label = real_label.cuda()

        if args.loss == 'wgan-gp':
            real_predict = discriminator(real_image, c=real_label, step=step, alpha=alpha)
            real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
            (-real_predict).backward()

        elif args.loss == 'r1':
            real_image.requires_grad = True
            real_scores = discriminator(real_image, c=real_label, step=step, alpha=alpha)
            real_predict = F.softplus(-real_scores).mean()
            real_predict.backward(retain_graph=True)

            grad_real = grad(
                outputs=real_scores.sum(), inputs=real_image, create_graph=True
            )[0]
            grad_penalty = (
                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
            ).mean()
            grad_penalty = 10 / 2 * grad_penalty
            grad_penalty.backward()
            if i % 10 == 0:
                grad_loss_val = grad_penalty.item()

        if args.contrastive:
            contr_loss_real = compute_contrastive(discriminator, real_image, contr_criterion, contr_transform, c=real_label,
                                                  step=step, alpha=alpha, weighting=args.contr_weighting)
            contr_loss_real.backward()

        if args.mixing and random.random() < 0.9:
            gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(
                4, b_size, args.code_size, device='cuda'
            ).chunk(4, 0)
            gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
            gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]

        else:
            gen_in1, gen_in2 = torch.randn(2, b_size, args.code_size, device='cuda').chunk(
                2, 0
            )
            gen_in1 = gen_in1.squeeze(0)
            gen_in2 = gen_in2.squeeze(0)

        fake_image = generator(gen_in1, step=step, alpha=alpha, task=task(real_label))
        fake_predict = discriminator(fake_image, c=real_label, step=step, alpha=alpha)

        if args.loss == 'wgan-gp':
            fake_predict = fake_predict.mean()
            fake_predict.backward()

            eps = torch.rand(b_size, 1, 1, 1).cuda()
            x_hat = eps * real_image.data + (1 - eps) * fake_image.data
            x_hat.requires_grad = True
            hat_predict = discriminator(x_hat, c=real_label, step=step, alpha=alpha)
            grad_x_hat = grad(
                outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
            )[0]
            grad_penalty = (
                (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
            ).mean()
            grad_penalty = 10 * grad_penalty
            grad_penalty.backward()
            if i % 10 == 0:
                grad_loss_val = grad_penalty.item()
                disc_loss_val = (-real_predict + fake_predict).item()

        elif args.loss == 'r1':
            fake_predict = F.softplus(fake_predict).mean()
            fake_predict.backward()
            if i % 10 == 0:
                disc_loss_val = (real_predict + fake_predict).item()

        if args.contrastive:
            fake_image = generator(gen_in1, step=step, alpha=alpha, task=task(real_label))
            contr_loss_fake = compute_contrastive(discriminator, fake_image, contr_criterion, contr_transform,
                                                  c=real_label, step=step, alpha=alpha, weighting=args.contr_weighting)
            contr_loss_fake.backward()

        d_optimizer.step()

        if (i + 1) % args.n_critic == 0:
            g_optimizer.zero_grad()

            requires_grad(generator, True, grad_map=grad_map_generator)
            requires_grad(discriminator, False)

            fake_image = generator(gen_in2, step=step, alpha=alpha, task=task(real_label))

            predict = discriminator(fake_image, c=real_label, step=step, alpha=alpha)

            if args.loss == 'wgan-gp':
                loss = -predict.mean()

            elif args.loss == 'r1':
                loss = F.softplus(-predict).mean()

            if i % 10 == 0:
                gen_loss_val = loss.item()

            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator.module, grad_map=grad_map_generator)
            accumulate(t_running, task.module)

            requires_grad(generator, False)
            requires_grad(discriminator, True, grad_map=grad_map_discriminator)

        if i % args.img_log_interval == 0:
            log_images(f'{run_dir}/fakes{i:06d}.png', alpha, args, dataset, resolution, real_image.size(0), step, g_running, t_running)

        if i % args.checkpoint_interval == 0 and i >= args.begin_checkpointing_at and not args.no_checkpointing:
            torch.save(
                {
                    'generator': generator.module.state_dict(),
                    'discriminator': discriminator.module.state_dict(),
                    # 'g_optimizer': g_optimizer.state_dict(),
                    # 'd_optimizer': d_optimizer.state_dict(),
                    'g_running': g_running.state_dict(),
                    'task_g': task.module.state_dict(),
                    't_running_g': t_running.state_dict(),
                },
                f'{run_dir}/{i:06d}.model',
            )

        state_msg = (
            f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
        )

        pbar.set_description(state_msg)
        if not args.no_tb:
            writer.add_scalar('1.loss/G', gen_loss_val, i)
            writer.add_scalar('1.loss/D', disc_loss_val, i)
            writer.add_scalar('1.loss/grad', grad_loss_val, i)
            if args.contrastive:
                writer.add_scalar('1.loss/BT_real', contr_loss_real, i)
                writer.add_scalar('1.loss/BT_fake', contr_loss_fake, i)
            writer.add_scalar('alpha', alpha, i)
            writer.add_scalar('resolution', 4 * 2 ** step, i)
            writer.add_scalar('batch_size', loader.batch_size, i)
            writer.add_scalar('lr/discriminator', d_optimizer.param_groups[0]['lr'], i)
            writer.add_scalar('lr/generator', g_optimizer.param_groups[0]['lr'], i)
            writer.add_scalar('lr/task', g_optimizer.param_groups[1]['lr'], i)
            writer.add_scalar('kimgs', num_imgs / 1000, i)


def get_args():
    parser = argparse.ArgumentParser(description='Hyper-Modulation')

    # GAN config
    parser.add_argument('path', type=str, help='path of specified dataset')
    parser.add_argument('--iterations', default=3_000_000, type=int, help='training iterations')
    parser.add_argument('--phase', type=int, default=600_000, help='number of samples used for each training phases')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--init_size', default=8, type=int, help='initial image size')
    parser.add_argument('--max_size', default=1024, type=int, help='max image size')
    parser.add_argument('--batch_default', default=32, type=int, help='batch size if no lr scheduling is activated')
    parser.add_argument('--n_critic', default=1, type=int, help='Number of critic iterations in a global iteration')
    parser.add_argument('--ckpt', default=None, type=str, help='load from previous checkpoints')
    parser.add_argument('--no_from_rgb_activate', action='store_true', help='use activate in from_rgb (original implementation)')
    parser.add_argument('--mixing', action='store_true', help='use mixing regularization')
    parser.add_argument('--loss', type=str, default='wgan-gp', choices=['wgan-gp', 'r1'], help='class of gan loss')
    parser.add_argument('--const_alpha', default=None, type=float, help='Whether to use a constant alpha')
    # Logging config
    parser.add_argument('--name', default=None, type=str, help='Name of the experiment')
    parser.add_argument('--run_dir', default='data/training-runs', type=str, help='')
    parser.add_argument('--no_tb', action='store_true', help='dont log in tensorboard')
    parser.add_argument('--no_checkpointing', action='store_true', help='dont save checkpoints')
    parser.add_argument('--begin_checkpointing_at', default=0, type=int, help='dont save model before a training point')
    parser.add_argument('--checkpoint_interval', default=10_000, type=int, help='how often to save the model')
    parser.add_argument('--img_log_interval', default=1_000, type=int, help='how often to generate images')
    # Dataset config
    parser.add_argument('--filter_classes', default=None, nargs='+', type=int, help='Use only n classes')
    parser.add_argument('--dataset_length', default=None, type=int, help='Constrain the number of samples')
    parser.add_argument('--dataset_class_samples', default=None, type=int, help='Constrain the number of samples')
    parser.add_argument('--dataset_random_class_sampling', action='store_true', help='Random sample the subset of class samples')
    # Hyper-modulation config
    parser.add_argument('--n_mlp_style', default=8, type=int, help='')
    parser.add_argument('--n_mlp_task', default=4, type=int, help='Task net depth')
    parser.add_argument('--code_size', default=512, type=int, help='')
    parser.add_argument('--task_size', default=64, type=int, help='task width')
    parser.add_argument('--finetune', action='store_true', help='Finetune underlying pretrained model')
    # Contrastive config
    parser.add_argument('--contrastive', action='store_true', help='Enable contrastive training')
    parser.add_argument('--contr_weighting', default=0.01, type=float, help='Contrastive training weighting in the final loss')
    parser.add_argument('--contr_feats', default=512, type=int, help='Dimensionality of features passed to the contrastive loss')

    return parser.parse_args()


def main(args):
    seed_everything(42)
    torch.backends.cudnn.benchmark = True
    contr_transform = None
    contr_criterion = None

    print(f'loss: {args.loss}')

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    if args.contrastive:
        contr_transform = ContrTransform(crop_resize=256)

    dataset = MultiResolutionDataset(args.path, transform, selected_classes=args.filter_classes, class_samples=args.dataset_class_samples, random_class_sampling=args.dataset_random_class_sampling, length=args.dataset_length)
    print(f'dataset len: {len(dataset)}')
    print(f'dataset classes: {dataset.num_classes}')
    args.num_classes = dataset.num_classes
    print(f'contrastive training: {args.contrastive}')

    generator = nn.DataParallel(StyledGenerator(code_dim=args.code_size, task_dim=args.task_size, n_mlp=args.n_mlp_style)).cuda()
    discriminator = nn.DataParallel(
        Discriminator(num_classes=args.num_classes, from_rgb_activate=not args.no_from_rgb_activate)
    ).cuda()
    task = nn.DataParallel(Task(args.task_size, n_mlp=args.n_mlp_task, num_labels=dataset.num_classes)).cuda()  # class network
    g_running = StyledGenerator(code_dim=args.code_size, task_dim=args.task_size, n_mlp=args.n_mlp_style).cuda()
    g_running.train(False)
    t_running = Task(args.task_size, n_mlp=args.n_mlp_task, num_labels=dataset.num_classes).cuda()
    t_running.train(False)
    if args.contrastive:
        contr_criterion = nn.DataParallel(BarlowTwins(num_feats=args.contr_feats, use_projector=False)).cuda()

    g_optimizer = optim.Adam(
        generator.module.generator.parameters(), lr=args.lr, betas=(0.0, 0.99)
    )
    g_optimizer.add_param_group(
        {
            'params': task.parameters(),
            'lr': args.lr * 0.01,
            'mult': 0.001,
        }
    )
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    if args.contrastive:
        d_optimizer.add_param_group({
            'params': contr_criterion.parameters(),
            'lr': args.lr,
        })

    accumulate(g_running, generator.module, 0)
    accumulate(t_running, task.module, 0)

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        if 'task_g' in ckpt:
            ckpt['task'] = ckpt['task_g']

        if 'task' in ckpt:  # load self-aligned checkpoint
            generator.module.load_state_dict(ckpt['generator'])
            # copy original class embedding for all new classes
            ckpt['task']['task.0.weight_orig'] = ckpt['task']['task.0.weight_orig'].repeat((dataset.num_classes, 1))
            task.module.load_state_dict(ckpt['task'])

            g_running.load_state_dict(ckpt['generator'])
            t_running.load_state_dict(ckpt['task'])
            del ckpt  # avoid OOM

            # load vanilla discriminator
            ckpt = torch.load('data/stylegan-256px-new.model')
            missing, unexpected = discriminator.module.load_state_dict(ckpt['discriminator'], strict=False)
            unexpected = [k for k in unexpected if not k.startswith('linear.')]
            assert len(unexpected) == 0
            missing = [k for k in missing if not k.startswith('final.')]
            assert len(missing) == 0
            del ckpt  # avoid OOM
        else:  # begin training from pretrained vanilla stylegan
            missing, unexpected = generator.module.load_state_dict(convert_generator(ckpt['generator']), strict=False)
            assert_loaded_keys(missing, unexpected)
            missing, unexpected = g_running.load_state_dict(convert_generator(ckpt['g_running']), strict=False)
            assert_loaded_keys(missing, unexpected)

            # load vanilla discriminator
            missing, unexpected = discriminator.module.load_state_dict(ckpt['discriminator'], strict=False)
            unexpected = [k for k in unexpected if not k.startswith('linear.')]
            assert len(unexpected) == 0
            missing = [k for k in missing if not k.startswith('final.')]
            assert len(missing) == 0
            del ckpt  # avoid OOM

        if not args.finetune:
            # can be stated here (after the optimizers declaration), as long as it's before the forward pass
            freeze_layers(generator, discriminator, g_running)

    if args.sched:
        args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 30, 128: 20, 256: 10}
    else:
        args.lr = {}
        args.batch = {}

    args.gen_sample = {512: (8, 4), 1024: (4, 2)}

    train(args, dataset, generator, discriminator, task, g_running, t_running, g_optimizer, d_optimizer,
          contr_transform, contr_criterion)


if __name__ == '__main__':
    main(get_args())
