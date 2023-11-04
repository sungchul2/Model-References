# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

from lightning_utilities import module_available
if module_available('lightning'):
    from lightning.pytorch import LightningDataModule
elif module_available('pytorch_lightning'):
    from pytorch_lightning import LightningDataModule

from sklearn.model_selection import KFold
from utils.utils import get_config_file, get_path, get_split, get_test_fnames, is_main_process, load_data
import os
import glob
from data_loading.dali_loader import fetch_dali_loader
from abc import ABC
from habana_frameworks.mediapipe import fn  # NOQA
from habana_frameworks.mediapipe.mediapipe import MediaPipe  # NOQA
from habana_frameworks.mediapipe.media_types import dtype as dt  # NOQA
from habana_frameworks.mediapipe.operators.cpu_nodes.cpu_nodes import media_function
import numpy as np

class DataModule(LightningDataModule if os.getenv('framework')=='PTL' else ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_imgs = []
        self.train_lbls = []
        self.val_imgs = []
        self.val_lbls = []
        self.test_imgs = []
        self.kfold = KFold(n_splits=self.args.nfolds, shuffle=True, random_state=12345)
        self.data_path = get_path(args)
        configs = get_config_file(self.args)

        if self.args.gpus:
            device_count = self.args.gpus
            device = 'gpu'
        elif self.args.hpus:
            device_count = self.args.hpus
            device = 'hpu'
        else:
            device_count = 1
            device = 'cpu'

        self.kwargs = {
            "dim": self.args.dim,
            "patch_size": configs["patch_size"],
            "seed": self.args.seed,
            "num_device": device_count,
            "num_workers": self.args.num_workers,
            "oversampling": self.args.oversampling,
            "benchmark": self.args.benchmark,
            "nvol": self.args.nvol,
            "train_batches": self.args.train_batches,
            "test_batches": self.args.test_batches,
            "meta": load_data(self.data_path, "*_meta.npy"),
            "device": device,
            "augment": self.args.augment,
            "set_aug_seed": self.args.set_aug_seed,
        }

    def setup(self, stage=None):
        imgs = load_data(self.data_path, "*_x.npy")
        lbls = load_data(self.data_path, "*_y.npy")

        self.test_imgs, self.kwargs["meta"] = get_test_fnames(self.args, self.data_path, self.kwargs["meta"])
        if self.args.exec_mode != "predict" or self.args.benchmark:
            train_idx, val_idx = list(self.kfold.split(imgs))[self.args.fold]
            self.train_imgs = get_split(imgs, train_idx)
            self.train_lbls = get_split(lbls, train_idx)
            self.val_imgs = get_split(imgs, val_idx)
            self.val_lbls = get_split(lbls, val_idx)
            if is_main_process():
                ntrain, nval = len(self.train_imgs), len(self.val_imgs)
                print(f"Number of examples: Train {ntrain} - Val {nval}")
        elif is_main_process():
            print(f"Number of test examples: {len(self.test_imgs)}")

    def train_dataloader(self):
        if self.args.habana_loader:
            from habana_dataloader import fetch_habana_unet_loader
            return fetch_habana_unet_loader(self.train_imgs, self.train_lbls, self.args.batch_size, "train", **self.kwargs)
        else:
            return fetch_dali_loader(self.train_imgs, self.train_lbls, self.args.batch_size, "train", **self.kwargs)

    def val_dataloader(self):
        if self.args.habana_loader:
            from habana_dataloader import fetch_habana_unet_loader
            return fetch_habana_unet_loader(self.val_imgs, self.val_lbls, 1, "eval", **self.kwargs)
        else:
            return fetch_dali_loader(self.val_imgs, self.val_lbls, 1, "eval", **self.kwargs)

    def test_dataloader(self):
        if self.args.habana_loader:
            from habana_dataloader import fetch_habana_unet_loader
            if self.kwargs["benchmark"]:
                return fetch_habana_unet_loader(self.train_imgs, self.train_lbls, self.args.val_batch_size, "test", **self.kwargs)
            return fetch_habana_unet_loader(self.test_imgs, None, 1, "test", **self.kwargs)
        else:
            if self.kwargs["benchmark"]:
                return fetch_dali_loader(self.train_imgs, self.train_lbls, self.args.val_batch_size, "test", **self.kwargs)
            return fetch_dali_loader(self.test_imgs, None, 1, "test", **self.kwargs)


class LiteHRNetDataModule(DataModule):
    def __init__(self, args):
        self.args = args
        self.train_imgs = []
        self.train_lbls = []
        self.val_imgs = []
        self.val_lbls = []
        self.test_imgs = []

        if self.args.gpus:
            device_count = self.args.gpus
            device = 'gpu'
        elif self.args.hpus:
            device_count = self.args.hpus
            device = 'hpu'
        else:
            device_count = 1
            device = 'cpu'

        self.kwargs = {
            "dim": self.args.dim,
            "seed": self.args.seed,
            "num_device": device_count,
            "num_workers": self.args.num_workers,
            "oversampling": self.args.oversampling,
            "benchmark": self.args.benchmark,
            "train_batches": self.args.train_batches,
            "test_batches": self.args.test_batches,
            "device": device,
            "augment": self.args.augment,
            "set_aug_seed": self.args.set_aug_seed,
        }

    def setup(self, stage=None):
        if self.args.exec_mode != "predict" or self.args.benchmark:
            self.train_imgs = sorted(glob.glob("/local/data/kitti/numpy_resized/train/*_img.npy"))
            self.train_lbls = sorted(glob.glob("/local/data/kitti/numpy_resized/train/*_mask.npy"))
            self.val_imgs = sorted(glob.glob("/local/data/kitti/numpy_resized/val/*_img.npy"))
            self.val_lbls = sorted(glob.glob("/local/data/kitti/numpy_resized/val/*_mask.npy"))
            if is_main_process():
                ntrain, nval = len(self.train_imgs), len(self.val_imgs)
                print(f"Number of examples: Train {ntrain} - Val {nval}")
        elif is_main_process():
            print(f"Number of test examples: {len(self.test_imgs)}")
            
    def train_dataloader(self):
        return fetch_habana_loader(
            args=self.args,
            imgs=self.train_imgs,
            masks=self.train_lbls,
            batch_size=self.args.batch_size,
            mode="train")

    def val_dataloader(self):
        return fetch_habana_loader(
            args=self.args,
            imgs=self.val_imgs,
            masks=self.val_lbls,
            batch_size=1,
            mode="val")
            
            
def fetch_habana_loader(args, imgs, masks, batch_size, mode, **kwargs):
    num_workers = kwargs.get("num_workers", 0)
    if num_workers != 0:
        print("Warning: num_workers is not supported by MediaDataLoader, ignoring num_workers: ", num_workers)

    device = "gaudi2"
    if mode == "train":
        num_threads = 1
    elif mode == "val":
        device = "cpu"
        num_threads = 2
    else:
        device = "cpu"
        num_threads = 2

    pipeline = LiteHRNetMediaPipe(
        device=device,
        imgs=imgs,
        masks=masks,
        batch_size=batch_size,
        mode=mode,
        prefetch_depth=3,
        num_instances=args.hpus,
        instance_id=int(os.getenv("LOCAL_RANK", "0")),
        num_threads=num_threads,
        seed=args.seed)

    if device == "cpu":
        from habana_frameworks.mediapipe.plugins.iterator_pytorch import CPUUnet3DPytorchIterator
        iterator = CPUUnet3DPytorchIterator(mediapipe=pipeline)
        return iterator
    else:
        from habana_frameworks.mediapipe.plugins.iterator_pytorch import HPUUnet3DPytorchIterator
        iterator = HPUUnet3DPytorchIterator(mediapipe=pipeline)
        return iterator


class LiteHRNetMediaPipe(MediaPipe):
    def __init__(
        self,
        device,
        imgs,
        masks,
        prefetch_depth,
        batch_size,
        mode="train",
        img_w=1241,
        img_h=375,
        resize_w=544,
        resize_h=544,
        crop_w=512,
        crop_h=512,
        seed=0,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        shuffle=True,
        drop_last=False,
        pad_remainder=True,
        num_instances=1,
        instance_id=0,
        num_threads=1,
    ):
        super().__init__(
            device=device,
            prefetch_depth=prefetch_depth,
            batch_size=batch_size,
            num_threads=num_threads,
            pipe_name=self.__class__.__name__)

        self.mode = mode
        if self.mode == "train":
            # Load
            self.input = fn.ReadNumpyDatasetFromDir(
                num_outputs=2,
                shuffle=shuffle,
                shuffle_across_dataset=shuffle,
                file_list=[imgs, masks],
                dtype=[dt.UINT8, dt.UINT8],
                seed=seed,
                num_readers=min(batch_size, 4),
                drop_remainder=drop_last,
                pad_remainder=pad_remainder,
                num_slices=num_instances,
                slice_index=instance_id) # CWHN
            
            # Resize
            ## CWHN -> NCWH
            self.pre_transpose_resize_img = fn.Transpose(permutation=[3, 0, 1, 2], tensorDim=4, dtype=dt.UINT8)
            self.pre_transpose_resize_mask = fn.Transpose(permutation=[3, 0, 1, 2], tensorDim=4, dtype=dt.UINT8)

            ## NCWH -> (N*C)WH1
            self.pre_reshape_resize_img = fn.Reshape(size=[batch_size*3, img_w, img_h, 1], tensorDim=4, layout='', dtype=dt.UINT8)
            self.pre_reshape_resize_mask = fn.Reshape(size=[batch_size*1, img_w, img_h, 1], tensorDim=4, layout='', dtype=dt.UINT8)
            
            self.resize_img = fn.Resize(mode=1, size1=resize_w, size2=resize_h, size3=1, dtype=dt.UINT8)
            self.resize_mask = fn.Resize(mode=0, size1=resize_w, size2=resize_h, size3=1, dtype=dt.UINT8)

            ## (N*C)WH1 -> NCWH
            self.post_reshape_resize_img = fn.Reshape(size=[batch_size, 3, resize_w, resize_h], tensorDim=4, layout='', dtype=dt.UINT8)
            self.post_reshape_resize_mask = fn.Reshape(size=[batch_size, 1, resize_w, resize_h], tensorDim=4, layout='', dtype=dt.UINT8)

            ## NCWH -> WHCN
            self.post_transpose_resize_img = fn.Transpose(permutation=[2, 3, 1, 0], tensorDim=4, dtype=dt.UINT8)
            self.post_transpose_resize_mask = fn.Transpose(permutation=[2, 3, 1, 0], tensorDim=4, dtype=dt.UINT8)

            # Random crop -> Fixed crop (WHCN)
            self.crop_img = fn.Crop(crop_w=crop_w, crop_h=crop_h, crop_pos_x=0.5, crop_pos_y=0.5, dtype=dt.UINT8)
            self.crop_mask = fn.Crop(crop_w=crop_w, crop_h=crop_h, crop_pos_x=0.5, crop_pos_y=0.5, dtype=dt.UINT8)

            # Random flip
            self.is_hflip = fn.MediaFunc(func=random_flip_func, shape=[batch_size], dtype=dt.UINT8, seed=seed, priv_params={"prob": 0.5})
            self.random_flip_img = fn.RandomFlip(horizontal=1, dtype=dt.UINT8)
            self.random_flip_mask = fn.RandomFlip(horizontal=1, dtype=dt.UINT8)

        else:
            # Load
            # self.input = fn.ReadNumpyDatasetFromDir(
            #     num_outputs=2,
            #     shuffle=False,
            #     shuffle_across_dataset=False,
            #     file_list=[imgs, masks],
            #     dtype=[dt.UINT8, dt.UINT8],
            #     seed=seed,
            #     drop_remainder=drop_last,
            #     pad_remainder=pad_remainder,
            #     num_slices=num_instances,
            #     slice_index=instance_id,
            #     device="cpu") # CWHN
            self.input_img = fn.ReadNumpyDatasetFromDir(
                num_outputs=1,
                shuffle=False,
                shuffle_across_dataset=False,
                file_list=imgs,
                dtype=dt.UINT8,
                seed=seed,
                drop_remainder=drop_last,
                pad_remainder=pad_remainder,
                num_slices=num_instances,
                slice_index=instance_id,
                device="cpu") # CWHN

            self.input_mask = fn.ReadNumpyDatasetFromDir(
                num_outputs=1,
                shuffle=False,
                shuffle_across_dataset=False,
                file_list=masks,
                dtype=dt.UINT8,
                seed=seed,
                drop_remainder=drop_last,
                pad_remainder=pad_remainder,
                num_slices=num_instances,
                slice_index=instance_id,
                device="cpu") # CWHN
            
            # Resize
            ## CWHN -> NCWH
            self.pre_transpose_resize_img = fn.Transpose(permutation=[3, 0, 1, 2], tensorDim=4, dtype=dt.UINT8)
            self.pre_transpose_resize_mask = fn.Transpose(permutation=[3, 0, 1, 2], tensorDim=4, dtype=dt.UINT8)

            ## NCWH -> (N*C)WH1
            self.pre_reshape_resize_img = fn.Reshape(size=[batch_size*3, img_w, img_h, 1], tensorDim=4, layout='', dtype=dt.UINT8)
            self.pre_reshape_resize_mask = fn.Reshape(size=[batch_size*1, img_w, img_h, 1], tensorDim=4, layout='', dtype=dt.UINT8)
            
            self.resize_img = fn.Resize(mode=1, size1=crop_w, size2=crop_h, size3=1, dtype=dt.UINT8)
            self.resize_mask = fn.Resize(mode=0, size1=crop_w, size2=crop_h, size3=1, dtype=dt.UINT8)

            ## (N*C)WH1 -> NCWH
            self.post_reshape_resize_img = fn.Reshape(size=[batch_size, 3, crop_w, crop_h], tensorDim=4, layout='', dtype=dt.UINT8)
            self.post_reshape_resize_mask = fn.Reshape(size=[batch_size, 1, crop_w, crop_h], tensorDim=4, layout='', dtype=dt.UINT8)

            ## NCWH -> WHCN
            self.post_transpose_resize_img = fn.Transpose(permutation=[2, 3, 1, 0], tensorDim=4, dtype=dt.UINT8)
            self.post_transpose_resize_mask = fn.Transpose(permutation=[2, 3, 1, 0], tensorDim=4, dtype=dt.UINT8)
            
        if self.mode == "train":
            # Normalize
            self.mean_node = fn.MediaConst(data=np.array(mean, dtype=np.float32), shape=[1, 1, 3], dtype=dt.FLOAT32)
            self.std_node = fn.MediaConst(data=np.array([1/s for s in std], dtype=np.float32), shape=[1, 1, 3], dtype=dt.FLOAT32)
            self.normalize = fn.CropMirrorNorm(crop_w=crop_w, crop_h=crop_h, dtype=dt.FLOAT32)
            
            # Cast mask
            self.cast_mask = fn.Cast(dtype=dt.INT32)
            
    def definegraph(self):
        if self.mode == "train":
            # load
            images, masks = self.input() # CWHN

            # resize
            images, masks = self.pre_transpose_resize_img(images), self.pre_transpose_resize_mask(masks) # NCWH
            images, masks = self.pre_reshape_resize_img(images), self.pre_reshape_resize_mask(masks) # (N*C)WH
            images, masks = self.resize_img(images), self.resize_mask(masks)
            images, masks = self.post_reshape_resize_img(images), self.post_reshape_resize_mask(masks) # NCWH
            images, masks = self.post_transpose_resize_img(images), self.post_transpose_resize_mask(masks) # WHCN

            # random crop
            images, masks = self.crop_img(images), self.crop_mask(masks) # WHCN

            # random flip
            is_hflip = self.is_hflip()
            images, masks = self.random_flip_img(images, is_hflip), self.random_flip_mask(masks, is_hflip) # WHCN
            
        else:
            # load
            images, masks = self.input_img(), self.input_mask() # CWHN

            # images, masks = self.pre_transpose_resize_img(images), self.pre_transpose_resize_mask(masks) # NCWH
            # images = self.pre_reshape_resize_img(images) # (N*C)WH
            # images = self.resize_img(images)
            # images = self.post_reshape_resize_img(images) # NCWH
            # images, masks = self.post_transpose_resize_img(images), self.post_transpose_resize_mask(masks) # WHCN
            
        if self.mode == "train":
            # normalize
            mean, std = self.mean_node(), self.std_node()
            images = self.normalize(images, mean, std) # WHCN
            
            masks = self.cast_mask(masks)

        return images, masks


class random_flip_func(media_function):
    """
    Class defining the random flip implementation.
    """
    def __init__(self, params):
        """
        Constructor method.
        :params params: dictionary of params conatining
                        shape: output shape of this class.
                        dtype: output dtype of this class.
                        seed: seed to be used for randomization.
        """
        self.np_shape = params['shape'][::-1]
        self.np_dtype = params['dtype']
        self.seed = params['seed']
        self.prob = params['priv_params']['prob']
        self.rng = np.random.default_rng(self.seed)

    def __call__(self):
        """
        Callable class method
        :returns : random flip values calculated per image.
        """
        a = self.rng.choice([0, 1], p=[1 - self.prob, self.prob], size=self.np_shape)
        a = np.array(a, dtype=self.np_dtype)
        return a
