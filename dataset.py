from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
import torch


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=8, selected_classes=None, class_samples=None, random_class_sampling=False, length=None, one_hot_label=False):
        self.path = path
        self.env = None
        self.open_lmdb()

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)
        self.filter_classes = None

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
            self.num_classes = txn.get('num_classes'.encode('utf-8'))

            if selected_classes is not None or class_samples is not None:
                if selected_classes is None:
                    selected_classes = list(range(int(self.num_classes.decode('utf-8')))) if self.num_classes is not None else [0]
                self.filter_classes = self.filtering(txn, selected_classes, class_samples, random_sampling=random_class_sampling)
                self.length = len(self.filter_classes)
                self.num_classes = str(len(selected_classes)).encode('utf-8')

        if self.num_classes is not None:
            self.num_classes = int(self.num_classes.decode('utf-8'))
        assert length is None or length <= self.length, f'There are not enough samples in the dataset. {length} asked, {self.length} in total.'
        if length is not None:
            self.length = length
        self.resolution = resolution
        self.transform = transform
        self.one_hot_label = one_hot_label

        self.close_lmdb()

    def open_lmdb(self):
        self.env = lmdb.open(
            self.path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def close_lmdb(self):
        self.env.close()
        self.env = None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.env is None:
            self.open_lmdb()

        if self.filter_classes is not None:
            index = self.filter_classes[index]

        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

            key = f'label-{str(index).zfill(5)}'.encode('utf-8')
            label = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        if label is not None:
            label = int(label.decode('utf-8'))
            if self.one_hot_label:
                label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=self.num_classes).float()

        return img, label

    @staticmethod
    def filtering(txn, selected_classes, class_samples, random_sampling=False):
        def get_label(txn, i):
            return int(txn.get(f'label-{str(i).zfill(5)}'.encode('utf-8')).decode('utf-8'))

        from collections import defaultdict
        import itertools
        import random
        class_ids = defaultdict(list)
        length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        # separate classes
        for i in range(length):
            class_ids[get_label(txn, i)].append(i)

        # drop unselected classes
        for i in list(class_ids.keys()):
            if i not in selected_classes:
                class_ids.pop(i)

        # reduce the samples per class
        for i in class_ids.keys():
            if random_sampling:
                class_ids[i] = random.sample(class_ids[i], k=class_samples)
            else:
                class_ids[i] = class_ids[i][:class_samples]

        # flatten
        return list(itertools.chain(*class_ids.values()))
