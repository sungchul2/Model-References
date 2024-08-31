import habana_frameworks.torch.hpu as hthpu
import habana_frameworks.torch.core
import habana_frameworks.torch.distributed.hccl
from habana_frameworks.mediapipe import fn  # NOQA
from habana_frameworks.mediapipe.mediapipe import MediaPipe  # NOQA
from habana_frameworks.mediapipe.plugins.iterator_pytorch import HPUUnet3DPytorchIterator
from habana_frameworks.mediapipe.media_types import dtype as dt  # NOQA
from habana_frameworks.mediapipe.operators.cpu_nodes.cpu_nodes import media_function
import habana_frameworks.torch.core as htcore


import glob
import os
import time
import datetime
import torch.multiprocessing as mp
import torch
import argparse
import torch.distributed as dist
from pytorch.npt import set_env_params, distribute_execution
from utils.utils import seed_everything, is_main_process
from types import SimpleNamespace
from typing import Optional, List

from pytorch.trainer import Trainer
import numpy as np

from mmseg.models import build_segmentor
from config.otx_litehrnet import config


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
            self.input = fn.ReadNumpyDatasetFromDir(
                num_outputs=2,
                shuffle=False,
                shuffle_across_dataset=False,
                file_list=[imgs, masks],
                dtype=[dt.UINT8, dt.UINT8],
                seed=seed,
                drop_remainder=drop_last,
                pad_remainder=pad_remainder,
                num_slices=num_instances,
                slice_index=instance_id,
                device="hpu") # CWHN
            # self.input_img = fn.ReadNumpyDatasetFromDir(
            #     num_outputs=1,
            #     shuffle=False,
            #     shuffle_across_dataset=False,
            #     file_list=imgs,
            #     dtype=dt.UINT8,
            #     seed=seed,
            #     drop_remainder=drop_last,
            #     pad_remainder=pad_remainder,
            #     num_slices=num_instances,
            #     slice_index=instance_id,
            #     device="cpu") # CWHN

            # self.input_mask = fn.ReadNumpyDatasetFromDir(
            #     num_outputs=1,
            #     shuffle=False,
            #     shuffle_across_dataset=False,
            #     file_list=masks,
            #     dtype=dt.UINT8,
            #     seed=seed,
            #     drop_remainder=drop_last,
            #     pad_remainder=pad_remainder,
            #     num_slices=num_instances,
            #     slice_index=instance_id,
            #     device="cpu") # CWHN
            
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
            images, masks = self.input() # CWHN
            # images, masks = self.input_img(), self.input_mask() # CWHN

            images, masks = self.pre_transpose_resize_img(images), self.pre_transpose_resize_mask(masks) # NCWH
            images = self.pre_reshape_resize_img(images) # (N*C)WH
            images = self.resize_img(images)
            images = self.post_reshape_resize_img(images) # NCWH
            images, masks = self.post_transpose_resize_img(images), self.post_transpose_resize_mask(masks) # WHCN
            
            # images, masks = self.transpose_img(images), self.transpose_mask(masks) # WHCN
            
        # if self.mode == "train":
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


##############################
######### DataModule #########
##############################
def fetch_habana_loader(args, imgs, masks, batch_size, mode, **kwargs):
    num_workers = kwargs.get("num_workers", 0)
    if num_workers != 0:
        print("Warning: num_workers is not supported by MediaDataLoader, ignoring num_workers: ", num_workers)

    device = "gaudi2"
    if mode == "train":
        num_threads = 1
    elif mode == "val":
        num_threads = 2
    else:
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

    iterator = HPUUnet3DPytorchIterator(mediapipe=pipeline)
    return iterator


class DataModule:
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage=None):
        if self.args.dataset == "kitti":
            self.train_imgs = sorted(glob.glob("/local/data/kitti/numpy_resized/train/*_img.npy"))
            self.train_masks = sorted(glob.glob("/local/data/kitti/numpy_resized/train/*_mask.npy"))
            self.val_imgs = sorted(glob.glob("/local/data/kitti/numpy_resized/val/*_img.npy"))
            self.val_masks = sorted(glob.glob("/local/data/kitti/numpy_resized/val/*_mask.npy"))
            if is_main_process():
                ntrain, nval = len(self.train_imgs), len(self.val_imgs)
                print(f"Number of examples: Train {ntrain} - Val {nval}")

    def train_dataloader(self):
        return fetch_habana_loader(
            args=self.args,
            imgs=self.train_imgs,
            masks=self.train_masks,
            batch_size=self.args.batch_size,
            mode="train")

    def val_dataloader(self):
        return fetch_habana_loader(
            args=self.args,
            imgs=self.val_imgs,
            masks=self.val_masks,
            batch_size=1,
            mode="val")
        
        
###############
### Metrics ###
###############
from models.metrics import stat_scores
class Dice:
    def __init__(self, nclass):
        self.nclass = nclass
        self.reset()

    def update(self, pred, target):
        self.n_updates += 1
        self.dice += self.compute_stats(pred, target)
    
    def reset(self):
        self.n_updates = torch.zeros(1, device=torch.device("hpu"))
        self.dice = torch.zeros((self.nclass,), device=torch.device("hpu"))

    def compute(self):
        return self.dice / self.n_updates

    def compute_stats(self, pred, target):
        scores = torch.zeros(self.nclass, device=pred.device, dtype=torch.float32)
        for i in range(self.nclass):
            if (target != i).all():
                # # no foreground class
                # _, _pred = torch.max(pred, 1)
                # score_add = torch.where((_pred != i).all(), 1, 0)
                # scores[i] += score_add
                continue
            _tp, _fp, _tn, _fn, _ = stat_scores(pred=pred, target=target, class_index=i)
            denom = (2 * _tp + _fp + _fn).to(torch.float)
            score_cls = torch.where(denom != 0.0, (2 * _tp).to(torch.float) / denom, 0.0)
            scores[i] += score_cls
        return scores


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ocr_lite_hrnet_18_mod2", choices=["ocr_lite_hrnet_s_mod2", "ocr_lite_hrnet_18_mod2"])
    parser.add_argument("--dataset", type=str, default="kitti")
    parser.add_argument("--hpus", type=int, default=1, help="Number of hpus")
    parser.add_argument("--run_lazy_mode", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--benchmark", type=bool, default=False)
    parser.add_argument("--exec_mode", type=str, default="train")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--gradient_clip", type=bool, default=True)
    parser.add_argument("--gradient_clip_norm", type=int, default=40)
    parser.add_argument("--is_autocast", type=bool, default=True)
    return parser.parse_args()


def train_pytorch(args, model):
    import tqdm
    from torch.optim import SGD, AdamW, Adam
    from habana_frameworks.torch.hpex.optimizers import FusedAdamW, FusedSGD

    data_module = DataModule(args)
    data_module.setup()
    train_dataloaders = [data_module.train_dataloader()]
    val_dataloaders = [data_module.val_dataloader()]
    # optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.99)
    # optimizer = FusedSGD(model.parameters(), lr=args.learning_rate, momentum=0.99)
    optimizer = FusedAdamW(model.parameters(), lr=args.learning_rate, eps=1e-08, weight_decay=args.weight_decay)
    # optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-08, weight_decay=args.weight_decay)
    dice = Dice(nclass=19)
    
    for epoch in range(1, args.max_epochs+1):
        model.train()
        pbar_train = tqdm.tqdm(train_dataloaders[0])
        for batch in pbar_train:
            optimizer.zero_grad(set_to_none=True)
            batch_size = batch["image"].shape[0]
            with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True):
                img, lbl = batch["image"], batch["label"]
                img, lbl = img.to(torch.device("hpu"), non_blocking=False), lbl.to(torch.device("hpu"), non_blocking=False)
                losses = model.forward_train(**{
                    "img": img,
                    "img_metas": [{
                        'ori_shape': (375, 1241, 3),
                        'pad_shape': (512, 512, 3),
                        'ori_filename': 'Dataset item index 142',
                        'filename': 'Dataset item index 142',
                        'scale_factor': np.array([1., 1., 1., 1.], dtype=np.float32),
                        'flip': True,
                        'img_norm_cfg': {
                            'mean': np.array([123.675, 116.28, 103.53], dtype=np.float32),
                            'std': np.array([58.395, 57.12, 57.375], dtype=np.float32),
                            'to_rgb': False
                        },
                        'flip_direction': 'horizontal',
                        'ignored_labels': np.array([], dtype=np.float64),
                        'img_shape': (512, 512, 3)
                    } for _ in range(batch_size)],
                    "gt_semantic_seg": lbl
                })
            loss = losses.get("decode.loss_ce")
            loss.backward()
            if args.gradient_clip == True:
                torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            htcore.mark_step()
            optimizer.step()
            htcore.mark_step()
            pbar_train.set_description(f"Epoch : {epoch:04d} | Loss : {loss:.4f}")
            
        model.eval()
        dice.reset()
        pbar_val = tqdm.tqdm(val_dataloaders[0])
        with torch.no_grad():
            for val_batch in pbar_val:
                with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True):
                    img, lbl = val_batch["image"], val_batch["label"]
                    img, lbl = img.to(torch.device("hpu"), non_blocking=False), lbl.to(torch.device("hpu"), non_blocking=False)
                    seg_pred = model.whole_inference(**{
                        "img": img,
                        "img_meta": [{
                            'ori_shape': (375, 1241, 3),
                            'pad_shape': (512, 512, 3),
                            'ori_filename': 'Dataset item index 142',
                            'filename': 'Dataset item index 142',
                            'scale_factor': np.array([1., 1., 1., 1.], dtype=np.float32),
                            'flip': False,
                            'img_norm_cfg': {
                                'mean': np.array([123.675, 116.28, 103.53], dtype=np.float32),
                                'std': np.array([58.395, 57.12, 57.375], dtype=np.float32),
                                'to_rgb': False
                            },
                            'ignored_labels': np.array([], dtype=np.float64),
                            'img_shape': (512, 512, 3)
                        }],
                        "rescale": True,
                    })
                htcore.mark_step()
                dice.update(seg_pred[0].argmax(axis=0), lbl.squeeze())
                cur_dice = dice.compute().clone().detach().cpu()
                cur_mdice = cur_dice.mean()
                pbar_val.set_description(f"Val mDice : {cur_mdice:.4f} " + str({f"class {i}": f"{d.data:.4f}" for i, d in enumerate(cur_dice)}))
            print(f"Epoch : {epoch:04d} | Val mDice : {cur_mdice:.4f} " + str({f"class {i}": f"{d.data:.4f}" for i, d in enumerate(cur_dice)}))

def main(args):
    print(args)
    os.environ["PT_HPU_LAZY_MODE"] = "1"
    hthpu.enable_dynamic_shape()
    
    seed_everything(seed=args.seed)
    torch.backends.cuda.deterministric = True
    torch.use_deterministic_algorithms(True)
    
    from mmseg.models import build_segmentor
    from train_litehrnet import config
    model = build_segmentor(config[args.model])
    model.to("hpu")
    
    start_time = time.time()
    train_pytorch(args, model)
    end_time = time.time()
    time_interval = end_time - start_time
    print("Total Training time ", datetime.timedelta(seconds=int(time_interval)))


if __name__ == "__main__":
    main(get_arguments())
