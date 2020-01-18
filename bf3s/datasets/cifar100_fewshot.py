import os
import pickle
import os.path
import numpy as np
from PIL import Image
import bf3s.utils as utils
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


_CIFAR_STD_PIXEL = [x / 255.0 for x in [63.0, 62.1, 66.7]]
_CIFAR_MEAN_PIXEL = [x / 255.0 for x in [125.3, 123.0, 113.9]]


class CIFAR100FewShot(data.Dataset):

    def __init__(self, phase="train", do_not_use_random_transf=False,
                 data_dir="./datasets/CIFAR", data_split_dir="./data/cifar-fs_splits"):
        assert phase in ("train", "val", "test")
        self.phase = phase
        self.name = "CIFAR100FewShot_" + phase

        normalize = transforms.Normalize(mean=_CIFAR_MEAN_PIXEL, std=_CIFAR_STD_PIXEL)

        if (self.phase == "test" or self.phase == "val") or do_not_use_random_transf:
            self.transform = transforms.Compose([lambda x: np.asarray(x), transforms.ToTensor(), normalize])
        else:
            self.transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                 transforms.RandomHorizontalFlip(), lambda x: np.asarray(x),
                                                 transforms.ToTensor(), normalize])
            pass

        cifar100_metadata_path = os.path.join(data_dir, "cifar-100-python", "meta")
        all_category_names = pickle.load(open(cifar100_metadata_path, "rb"))["fine_label_names"]

        def read_categories(filename):
            with open(filename) as f:
                categories = f.readlines()
            categories = [x.strip() for x in categories]
            return categories

        def get_label_ids(category_names):
            label_ids = [all_category_names.index(cname) for cname in category_names]
            return label_ids

        train_category_names = read_categories(os.path.join(data_split_dir, "train.txt"))
        val_category_names = read_categories(os.path.join(data_split_dir, "val.txt"))
        test_category_names = read_categories(os.path.join(data_split_dir, "test.txt"))

        train_category_ids = get_label_ids(train_category_names)
        val_category_ids = get_label_ids(val_category_names)
        test_category_ids = get_label_ids(test_category_names)

        print(f"Loading CIFAR-100 FewShot dataset - phase {phase}")

        if self.phase == "train":
            self.data_train = datasets.__dict__["CIFAR100"](data_dir, train=True,
                                                            download=True, transform=self.transform)
            self.labels = self.data_train.targets
            self.images = self.data_train.data

            self.label2ind = utils.build_label_index(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = train_category_ids
            self.num_cats_base = len(self.labelIds_base)
        elif self.phase == "val" or self.phase == "test":
            self.data_train = datasets.__dict__["CIFAR100"](data_dir, train=True,
                                                            download=True, transform=self.transform)
            labels_train = self.data_train.targets
            images_train = self.data_train.data
            label2ind_train = utils.build_label_index(labels_train)
            self.labelIds_novel = (val_category_ids if (self.phase == "val") else test_category_ids)

            labels_novel = []
            images_novel = []
            for label_id in self.labelIds_novel:
                indices = label2ind_train[label_id]
                images_novel.append(images_train[indices])
                labels_novel += [labels_train[index] for index in indices]
                pass
            images_novel = np.concatenate(images_novel, axis=0)
            assert images_novel.shape[0] == len(labels_novel)

            self.data_test = datasets.__dict__["CIFAR100"](data_dir, train=False,
                                                           download=True, transform=self.transform)
            labels_test = self.data_test.targets
            images_test = self.data_test.data
            label2ind_test = utils.build_label_index(labels_test)
            self.labelIds_base = train_category_ids

            labels_base = []
            images_base = []
            for label_id in self.labelIds_base:
                indices = label2ind_test[label_id]
                images_base.append(images_test[indices])
                labels_base += [labels_test[index] for index in indices]
                pass

            images_base = np.concatenate(images_base, axis=0)
            assert images_base.shape[0] == len(labels_base)

            self.images = np.concatenate([images_base, images_novel], axis=0)
            self.labels = labels_base + labels_novel
            assert self.images.shape[0] == len(self.labels)

            self.num_cats_base = len(self.labelIds_base)
            self.num_cats_novel = len(self.labelIds_novel)
            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert len(intersection) == 0

            self.label2ind_base = utils.build_label_index(labels_base)
            assert len(self.label2ind_base) == self.num_cats_base

            self.label2ind_novel = utils.build_label_index(labels_novel)
            assert len(self.label2ind_novel) == self.num_cats_novel

            self.label2ind = utils.build_label_index(self.labels)
            assert len(self.label2ind) == self.num_cats_novel + self.num_cats_base
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
        else:
            raise ValueError(f"Not valid phase {self.phase}")

        pass

    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
            pass
        return img, label

    def __len__(self):
        return len(self.images)

    pass
