"""Utilities to fiddle with the models. Convert a pre-trained StyleGAN. Freeze all but hypernetwork layers."""
from math import sqrt


def conv(num, layer=None, parameter='weight', fused=False):
    return f'conv{num}.{"" if layer is None else str(layer) + "."}{"" if fused else "conv."}{parameter}'


def equalization_norm(w):
    fan_in = w.data.size(1) * w.data[0][0].numel()
    w = w * sqrt(2 / fan_in)
    return w


def normalize_conv2d_weight(w):
    w = equalization_norm(w)
    w_mu = w.mean((2, 3), keepdim=True)
    w_std = w.std((2, 3), keepdim=True)
    w = (w - w_mu) / w_std
    return w


def normalize_conv2d_bias(w):
    w_mu = w.mean()
    w_std = w.std()
    w = (w - w_mu) / w_std
    return w


def rename_and_norm(state_new, k, parameter, fused=False):
    minus_layers = 1 if fused else 2

    if parameter == 'weight':
        state_new[k.rsplit('.', minus_layers)[0] + '.W'] = normalize_conv2d_weight(state_new.pop(k))
    if parameter == 'bias':
        state_new[k.rsplit('.', minus_layers)[0] + '.b'] = normalize_conv2d_bias(state_new.pop(k))


def convert_generator(state):
    state_new = state.copy()  # should be a shallow copy. I.e. tensors in the dict values should be referenced.
    for k, v in state.items():
        parts = k.split('.')
        module = parts[0]

        if module == 'generator':
            module_gen = parts[1]
            if module_gen == 'progression':
                s = k.split('.', 3)[-1]

                if s == conv(1, 0, parameter='weight', fused=True):
                    rename_and_norm(state_new, k, 'weight', fused=True)

                if s == conv(1, 0, parameter='bias', fused=True):
                    rename_and_norm(state_new, k, 'bias', fused=True)

                if s == conv(1, 1, parameter='weight_orig'):
                    rename_and_norm(state_new, k, 'weight')

                if s == conv(1, 1, parameter='bias'):
                    rename_and_norm(state_new, k, 'bias')

                if s == conv(2, parameter='weight_orig'):
                    rename_and_norm(state_new, k, 'weight')

                if s == conv(2, parameter='bias'):
                    rename_and_norm(state_new, k, 'bias')

    state_new['generator.progression.0.conv1.0.input'] = state_new.pop('generator.progression.0.conv1.input')

    return state_new


def assert_loaded_keys(missing, unexpected):
    def check(k):
        return not (
                k.endswith('gamma') or k.endswith('beta') or k.endswith('bias_beta')
                or k.find('task') >= 0 or k.startswith('style.1.linear.') or k.startswith('final.')
        )
    missing = [k for k in missing if check(k)]
    assert len(missing) == 0
    unexpected = [k for k in unexpected if not k.startswith('linear.')]
    assert len(unexpected) == 0


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def freeze_layers(*models):
    """Freeze selected generator and discriminator weights"""
    from model.hyper_mod import ConstantInput, StyledGenerator, Discriminator
    from torch.nn import DataParallel
    from torch.nn.parallel.distributed import DistributedDataParallel

    for model in models:
        if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
            model = model.module

        if isinstance(model, StyledGenerator):
            for styled_block in model.generator.progression:
                if isinstance(styled_block.conv1[0], ConstantInput):
                    freeze(styled_block.conv1)

                # no need to freeze conv1 nor conv2 bc parameters have been already converted to buffers
                freeze(styled_block.noise1)
                freeze(styled_block.adain1)
                freeze(styled_block.noise2)
                freeze(styled_block.adain2)

            for to_rgb_i in model.generator.to_rgb:
                freeze(to_rgb_i)

            assert len(model.style) == 8 * 2 + 1  # 8 layers
            freeze(model.style)
        elif isinstance(model, Discriminator):
            for from_rgb_i in model.from_rgb:
                freeze(from_rgb_i)
        elif model is not None:
            print(type(model))
            raise ValueError
