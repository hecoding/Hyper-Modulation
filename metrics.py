import argparse
from pathlib import Path
from typing import Optional, Tuple, Union
from io import BytesIO

from tqdm import tqdm, trange
import numpy as np
import lmdb
import torch
from torchvision.transforms import ToTensor
from dataset import MultiResolutionDataset
from torch.utils.data import DataLoader
from piq.feature_extractors import InceptionV3
from piq import FID, KID

from metric_impl.pr import PR
from metric_impl.dc import DC
from utils import load_hparams, load_checkpoint


def to_bytes(img):
    buffer = BytesIO()
    torch.save(img, buffer)
    return buffer.getvalue()


def get_z(num_samples=1, code_dim=512, device='cuda'):
    return torch.randn(num_samples, code_dim, device=device)


def clamp_img(img: torch.Tensor, range: Optional[Tuple[int, int]] = (-1, 1)) -> None:
    img.clamp_(range[0], range[1])


def featurize(extractor, imgs):
    with torch.no_grad():
        feats = extractor(imgs)
    return feats[0].squeeze(-1).squeeze(-1)


def pick_samples(orig_size, selection_size, method):
    assert method in ['first', 'random']
    if selection_size is None:
        return torch.arange(orig_size)
    if selection_size > orig_size:
        selection_size = orig_size

    if method == 'first':
        return torch.arange(selection_size)
    elif method == 'random':
        return torch.from_numpy(np.random.choice(orig_size, size=selection_size, replace=False))


def load_feats(path, classes=1, load_n=None, sampling_method='first', device='cpu'):
    """
    Load precomputed inception features from an LMDB file.

    :param path: Path to the features.
    :param classes: Int with the number of classes to load or iterable consisting of class indices.
    :param load_n: Will load N samples per class. To pick which ones, see sampling_method param.
    :param sampling_method: Pick the first N samples for each class, or pick N random ones.
    :param device: Which device to load features to.
    """
    if isinstance(classes, int):
        classes_iter = range(classes)
    elif isinstance(classes, tuple) or isinstance(classes, list):
        classes_iter = classes
    else:
        raise ValueError

    feats = {}
    with lmdb.open(path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False) as env:
        for c in classes_iter:
            with env.begin(write=False) as txn:
                key = f'{c}'.encode('utf-8')
                feat = txn.get(key)
                feat = torch.load(BytesIO(feat), map_location=device)
                # in the loop bc classes can have different number of samples
                idx = pick_samples(orig_size=feat.shape[0], selection_size=load_n, method=sampling_method)
                # clone for keeping only the slice in memory, not the whole underlying tensor storage
                feats[c] = feat[idx].clone()

    return feats


def generate(z, model, task, classes, step=6, device='cuda'):
    if isinstance(classes, tuple) or isinstance(classes, list) or isinstance(classes, range):
        classes = torch.tensor(classes, device=device)
    elif isinstance(classes, torch.Tensor):
        if classes.device != device:
            classes = classes.to(device)
    else:
        raise ValueError("Wrong classes type")

    with torch.no_grad():
        task_input = task(classes)
        output = model(z, task=task_input, step=step)
    return output


def generate_class(model, task, class_i, num_samples=10, code_dim=512, device='cuda'):
    if isinstance(class_i, torch.Tensor):
        conditioning = class_i
    else:
        conditioning = [class_i] * num_samples
    z = get_z(num_samples, code_dim=code_dim, device=device)
    return generate(z, model, task, conditioning, device=device)


def generate_metrics_by_class(generator, task, metrics: dict, real_feats: Union[torch.Tensor, dict], num_classes=1, num_samples=10_000, batch_size=2, code_dim=512, feats_dim=2048, device='cuda'):
    metric_results_classes = {}
    feature_extractor = InceptionV3(normalize_input=False)  # already (-1, 1) range from generator output
    feature_extractor.to(device)
    feature_extractor.eval()

    for c in trange(num_classes, desc='Classes', leave=False, disable=num_classes == 1):
        metric_results = {}
        fake_feats = torch.empty((num_samples, feats_dim), device=device)
        last_batch_size = num_samples % batch_size

        for idx, i in enumerate(trange(0, num_samples, batch_size, desc='Samples', leave=False)):
            current_bs = last_batch_size if idx == num_samples // batch_size else batch_size
            imgs = generate_class(generator, task, c, current_bs, code_dim, device=device)
            clamp_img(imgs)  # sometimes images go beyond their range, thus Inception extractor will complain
            fake_feats[i:i + current_bs] = featurize(feature_extractor, imgs)

        for metric_name, metric in metrics.items():
            fake_i = None if metric_name != 'kid' else real_feats[c].shape[0]  # kid needs reals >= fakes
            score = metric(x_features=fake_feats[:fake_i], y_features=real_feats[c])  # predicted_feats, target_feats
            if isinstance(score, tuple):
                score = {f'{i}': s.item() for i, s in enumerate(score)}
            elif isinstance(score, torch.Tensor):
                score = score.item()
            else:
                raise RuntimeError('Unknown metric score type')
            metric_results[metric_name] = score

        metric_results_classes[c] = metric_results

    return metric_results_classes


def generate_metrics_mixed_class(generator, task, metrics: dict, real_feats: Union[torch.Tensor, dict], num_classes=1, num_samples=10_000, batch_size=2, code_dim=512, feats_dim=2048, device='cuda'):
    feature_extractor = InceptionV3(normalize_input=False)  # already (-1, 1) range from generator output
    feature_extractor.to(device)
    feature_extractor.eval()

    metric_results = {}
    real_feats = real_feats[0]
    fake_feats = torch.empty((num_samples, feats_dim), device=device)
    last_batch_size = num_samples % batch_size

    for idx, i in enumerate(trange(0, num_samples, batch_size, desc='Samples', leave=False)):
        current_bs = last_batch_size if idx == num_samples // batch_size else batch_size
        c = torch.from_numpy(np.random.choice(num_classes, size=current_bs)).to(device)
        imgs = generate_class(generator, task, c, current_bs, code_dim, device=device)
        clamp_img(imgs)  # sometimes images go beyond their range, thus Inception extractor will complain
        fake_feats[i:i + current_bs] = featurize(feature_extractor, imgs)

    for metric_name, metric in metrics.items():
        fake_i = None if metric_name != 'kid' else real_feats.shape[0]  # kid needs reals >= fakes
        score = metric(x_features=fake_feats[:fake_i], y_features=real_feats)  # predicted_feats, target_feats
        if isinstance(score, tuple):
            score = {f'{i}': s.item() for i, s in enumerate(score)}
        elif isinstance(score, torch.Tensor):
            score = score.item()
        else:
            raise RuntimeError('Unknown metric score type')
        metric_results[metric_name] = score

    return {0: metric_results}


def compute_feats_from_dataset(db_path, batch_size=1,  output_path='output_lmdb', device='cuda'):
    feats_dim = 2048
    resolution = 256
    feature_extractor = InceptionV3(normalize_input=True)
    feature_extractor.to(device)
    feature_extractor.eval()

    dataset = MultiResolutionDataset(db_path, transform=ToTensor(), resolution=resolution)
    num_classes = dataset.num_classes
    with lmdb.open(output_path, map_size=1024 ** 4, readahead=False) as env:
        for c in trange(num_classes, desc='Classes'):
            dataset = MultiResolutionDataset(db_path, transform=ToTensor(), resolution=resolution, selected_classes=[c])
            feats_c = torch.empty((dataset.length, feats_dim), device=device)
            for i, (imgs, labels) in enumerate(tqdm(DataLoader(dataset, batch_size=batch_size, num_workers=4), desc='Batches', leave=False)):
                feats_c[i * batch_size:i * batch_size + imgs.size(0)] = featurize(feature_extractor, imgs.to(device))
            with env.begin(write=True) as txn:
                key = f'{c}'.encode('utf-8')
                txn.put(key, to_bytes(feats_c.cpu()))


def compute_metrics_per_ckpt(args):
    """
    Compute specified metrics class-wise or for all classes and save results in a pickle. It will be run for each
    checkpoint it finds, creating a lock file for parallel computation (cheap but useful).
    """
    import pickle
    mix_dir = 'mixed' if args.classes == 'interclass' else ''
    metrics = {'fid': FID(), 'kid': KID(), 'pr': PR(), 'dc': DC()}
    hparams = load_hparams(args.checkpoint_path)
    if not hasattr(hparams, 'num_classes'):
        hparams.num_classes = 1
    # 1 class and random if mix classes bc the precomputed feats dataset supposed to be passed has all classes in
    # the first class, so getting the first 10k will likely just get all samples from one/two classes
    real_feats = load_feats(args.precomputed_feats, classes=1 if args.classes == 'interclass' else hparams.num_classes,
                            load_n=10_000, sampling_method='random' if args.classes == 'interclass' else 'first',
                            device=args.device)

    checkpoint_folder = Path(args.checkpoint_path)
    if args.classes == 'interclass':
        (checkpoint_folder / mix_dir).mkdir(exist_ok=True)

    for epoch in tqdm(list(sorted(checkpoint_folder.glob('*.model'), reverse=True)), desc='Epoch'):
        output_score = epoch.parent / mix_dir / (epoch.stem + '.metric')
        epoch_lock = epoch.parent / mix_dir / (epoch.stem + '.lock')
        if output_score.exists() or epoch_lock.exists():
            continue
        epoch_lock.touch(exist_ok=False)

        args.checkpoint_path = str(epoch)
        try:
            generator, task = load_checkpoint(args, hparams)
        except Exception as e:
            epoch_lock.unlink()
            raise e

        if args.classes == 'interclass':
            scores = generate_metrics_mixed_class(generator, task, metrics, real_feats,
                                                  num_classes=hparams.num_classes, num_samples=args.num_samples,
                                                  batch_size=args.batch_size, code_dim=hparams.code_size,
                                                  feats_dim=args.feats_dim, device=args.device)
        elif args.classes == 'intraclass':
            scores = generate_metrics_by_class(generator, task, metrics, real_feats,
                                               num_classes=hparams.num_classes, num_samples=args.num_samples,
                                               batch_size=args.batch_size, code_dim=hparams.code_size,
                                               feats_dim=args.feats_dim, device=args.device)
        else:
            raise ValueError

        with open(output_score, 'wb') as f:
            pickle.dump(scores, f)

        if args.remove_checkpoint_afterwards:
            epoch.unlink()
        epoch_lock.unlink()


def compute_metrics_biggan(imgs_path, num_samples=10_000, batch_size=2, feats_dim=2048, device='cuda'):
    """
    Read BigGAN generations (npz file) and compute metrics.
    """
    metrics = {'fid': FID(), 'kid': KID(), 'pr': PR(), 'dc': DC()}
    np_imgs = np.load(imgs_path)
    print(f'Loading {imgs_path}')
    y = np_imgs['y']
    num_classes = len(np.unique(y))
    assert np.array_equal(np.arange(num_classes), np.unique(y)), f'Discontinuous number of classes in {imgs_path}'

    x = {}
    for c in trange(num_classes, desc='Loading classes'):
        idcs = np.where(y == c)[0][:num_samples]
        if len(idcs) < num_samples:
            print(f'Class {c} has less number of samples than requested ({num_samples})')
        x[c] = np_imgs['x'][idcs]

    real_feats = load_feats(args.precomputed_feats, classes=num_classes, load_n=10_000, device=args.device)

    metric_results_classes = {}
    feature_extractor = InceptionV3(normalize_input=True)  # [0, 1] to [-1, 1]
    feature_extractor.to(device)
    feature_extractor.eval()

    for c in trange(num_classes, desc='Classes', leave=False, disable=num_classes == 1):
        metric_results = {}
        fake_feats = torch.empty((num_samples, feats_dim), device=device)
        last_batch_size = num_samples % batch_size

        for idx, i in enumerate(trange(0, num_samples, batch_size, desc='Samples', leave=False)):
            current_bs = last_batch_size if idx == num_samples // batch_size else batch_size
            imgs = torch.from_numpy(x[c][i:i + current_bs]).to(device) / 255
            fake_feats[i:i + current_bs] = featurize(feature_extractor, imgs)

        for metric_name, metric in metrics.items():
            score = metric(real_feats[c], fake_feats)
            if isinstance(score, tuple):
                score = {f'{i}': s.item() for i, s in enumerate(score)}
            elif isinstance(score, torch.Tensor):
                score = score.item()
            else:
                raise RuntimeError('Unknown metric score type')
            metric_results[metric_name] = score

        metric_results_classes[c] = metric_results

    return metric_results_classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str, help='Path of specified dataset')
    parser.add_argument('--output_path', type=str, default='Output LMDB for precomputed features')
    parser.add_argument('--precomputed_feats', type=str, default=None, help='If the function needs it')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_samples', type=int, default=10_000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--feats_dim', type=int, default=2048)
    parser.add_argument('--remove_checkpoint_afterwards', action='store_true', help='Remove computing metrics')
    parser.add_argument('--operation', type=str, default='metrics', choices=['extract-feats', 'metrics', 'metrics-biggan'])
    parser.add_argument('--classes', type=str, default='intraclass', choices=['interclass', 'intraclass'], help='Evaluation regarding conditioning')
    args = parser.parse_args()

    if args.operation == 'extract-feats':
        compute_feats_from_dataset(args.checkpoint_path, batch_size=args.batch_size, output_path=args.output_path, device=args.device)
    elif args.operation == 'metrics':
        compute_metrics_per_ckpt(args)
    elif args.operation == 'metrics-biggan':
        print(compute_metrics_biggan(args.checkpoint_path, batch_size=args.batch_size))
