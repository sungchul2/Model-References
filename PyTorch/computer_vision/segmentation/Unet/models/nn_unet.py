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


import os
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD

from skimage.transform import resize
from torch_optimizer import RAdam
from utils.utils import (
    flip,
    get_dllogger,
    get_path,
    get_test_fnames,
    get_tta_flips,
    get_unet_params,
    is_main_process,
    layout_2d,
    mark_step,
    get_device,
    get_device_str,
    get_device_data_type
)

from models.loss import Loss
from models.metrics import Dice
from models.unet import UNet
from config.otx_litehrnet import config
from mmseg.models import build_segmentor

from lightning_utilities import module_available
if module_available("lightning"):
    import lightning.pytorch as pl
elif module_available("pytorch_lightning"):
    import pytorch_lightning as pl

class NNUnet(pl.LightningModule if os.getenv('framework')=='PTL' else nn.Module):
    def __init__(self, args):
        super(NNUnet, self).__init__()
        self.validation_step_outputs = []
        self.args = args
        if not hasattr(self.args, "drop_block"):  # For backward compability
            self.args.drop_block = False
        if hasattr(self, 'save_hyperparameters'): #Pytorch and PTL compatibility
            self.save_hyperparameters()
        self.build_nnunet()
        self.loss = Loss(self.args.focal)
        self.dice = Dice(self.n_class)
        if hasattr(self.args, 'use_torch_compile') and self.args.use_torch_compile:
            self.loss = torch.compile(self.loss, backend="aot_hpu_training_backend")
            self.dice = torch.compile(self.dice, backend="aot_hpu_training_backend")
        self.first = True
        self.best_sum = 0
        self.best_sum_epoch = 0
        self.best_dice = self.n_class * [0]
        self.best_epoch = self.n_class * [0]
        self.best_sum_dice = self.n_class * [0]
        self.learning_rate = args.learning_rate
        self.tta_flips = get_tta_flips(args.dim)
        self.test_idx = 0
        self.test_imgs = []
        if self.args.exec_mode in ["train", "evaluate"]:
            self.dllogger = get_dllogger(args.results)
        self.window_size = 20
        self.train_loss_valid_end = 0
        self.train_loss = torch.zeros(self.window_size, device=torch.device(get_device_str(self.args)))

    def forward(self, img):
        with torch.autocast(device_type=get_device_str(self.args), dtype=get_device_data_type(self.args), enabled=self.args.is_autocast):
            if self.args.benchmark:
                if self.args.dim == 2 and self.args.data2d_dim == 3:
                    img = layout_2d(img, None)
                return self.model(img)
            return self.tta_inference(img) if self.args.tta else self.do_inference(img)

    def training_step(self, batch, batch_idx):
        with torch.autocast(device_type=get_device_str(self.args), dtype=get_device_data_type(self.args), enabled=self.args.is_autocast):
            img, lbl = self.get_train_data(batch)
            pred = self.model(img)
            loss = self.compute_loss(pred, lbl)
        # WA for https://github.com/Lightning-AI/lightning/issues/17251
        # TBD: move to use of trochmetrics==v1.0.0 for following calc
        if batch_idx  % self.args.progress_bar_refresh_rate == 0:
            self.train_loss = torch.cat((torch.tensor([loss], device=self.train_loss.device), self.train_loss[:-1]))
            if self.train_loss_valid_end < self.window_size:
                self.train_loss_valid_end += 1
                mask = (torch.arange(len(self.train_loss)) >= 0) & (torch.arange(len(self.train_loss)) <= self.train_loss_valid_end)
                moving_average_loss = torch.mean(self.train_loss[mask])
            else:
                moving_average_loss = torch.mean(self.train_loss)
            self.log("loss", moving_average_loss, prog_bar=True)
        return loss


    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def on_after_backward(self):
        mark_step(self.args.run_lazy_mode)

    def validation_step(self, batch, batch_idx):
        with torch.autocast(device_type=get_device_str(self.args), dtype=get_device_data_type(self.args), enabled=self.args.is_autocast):
            if os.getenv('framework') == 'NPT': #Pytorch and PTL compatibility
                self.current_epoch = self.trainer.current_epoch
            if self.current_epoch < self.args.skip_first_n_eval:
                return None
            img, lbl = batch["image"], batch["label"]
            if self.args.hpus:
                img, lbl = img.to(torch.device("hpu"), non_blocking=False), lbl.to(torch.device("hpu"), non_blocking=False)
            pred = self.forward(img)
            if self.args.dim == 3:
                mark_step(self.args.run_lazy_mode)
            #Calculating dice update before calculating loss as dice update has blocking calls of "if conditions" to fetch tensors to CPU
            self.dice.update(pred, lbl[:, 0])
            loss = self.loss(pred, lbl)
            mark_step(self.args.run_lazy_mode)
            self.validation_step_outputs.append(loss)
            return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        if self.args.exec_mode == "evaluate":
            return self.validation_step(batch, batch_idx)
        if batch_idx == 0:
            print("Start test")
            if self.args.hpus and self.args.inference_mode == "graphs" and self.first:
                from habana_frameworks.torch.hpu.graphs import wrap_in_hpu_graph
                self.model = wrap_in_hpu_graph(self.model)
                self.first = False
        img = batch["image"]
        if self.args.hpus and not self.args.benchmark and self.args.dim == 2:
            img = img.cpu()

        pred = self.forward(img)
        mark_step(self.args.run_lazy_mode)
        if (batch_idx == self.args.test_batches -1 or self.args.measurement_type == 'latency') and self.args.benchmark:
            _ = pred.cpu()

        if self.args.save_preds:
            with torch.autocast(device_type=get_device_str(self.args), dtype=get_device_data_type(self.args), enabled=self.args.is_autocast):
                meta = batch["meta"][0].cpu().detach().numpy()
                original_shape = meta[2]
                min_d, max_d = meta[0, 0], meta[1, 0]
                min_h, max_h = meta[0, 1], meta[1, 1]
                min_w, max_w = meta[0, 2], meta[1, 2]

                final_pred = torch.zeros((1, pred.shape[1], *original_shape), device=img.device)
                final_pred[:, :, min_d:max_d, min_h:max_h, min_w:max_w] = pred
                final_pred = nn.functional.softmax(final_pred, dim=1)
                final_pred = final_pred.squeeze(0).cpu().detach().numpy()

                if not all(original_shape == final_pred.shape[1:]):
                    class_ = final_pred.shape[0]
                    resized_pred = np.zeros((class_, *original_shape))
                    for i in range(class_):
                        resized_pred[i] = resize(
                            final_pred[i], original_shape, order=3, mode="edge", cval=0, clip=True, anti_aliasing=False
                        )
                    final_pred = resized_pred

                self.save_mask(final_pred)


    def build_nnunet(self):
        in_channels, n_class, kernels, strides, self.patch_size = get_unet_params(self.args)
        self.n_class = n_class - 1
        self.model = UNet(
            in_channels=in_channels,
            n_class=n_class,
            kernels=kernels,
            strides=strides,
            dimension=self.args.dim,
            residual=self.args.residual,
            attention=self.args.attention,
            drop_block=self.args.drop_block,
            normalization_layer=self.args.norm,
            negative_slope=self.args.negative_slope,
            deep_supervision=self.args.deep_supervision,
        )
        if hasattr(self.args, 'use_torch_compile') and self.args.use_torch_compile:
            self.model = torch.compile(self.model, backend="aot_hpu_training_backend")
        if is_main_process():
            print(f"Filters: {self.model.filters},\nKernels: {kernels}\nStrides: {strides}")

    def compute_loss(self, preds, label):
        if self.args.deep_supervision:
            loss = self.loss(preds[0], label)
            for i, pred in enumerate(preds[1:]):
                downsampled_label = nn.functional.interpolate(label, pred.shape[2:])
                loss += 0.5 ** (i + 1) * self.loss(pred, downsampled_label)
            c_norm = 1 / (2 - 2 ** (-len(preds)))
            return c_norm * loss
        return self.loss(preds, label)

    def do_inference(self, image):
        if self.args.dim == 3:
            return self.sliding_window_inference(image)
        if self.args.data2d_dim == 2:
            return self.model(image)
        if self.args.exec_mode == "predict":
            return self.inference2d_test(image)
        return self.inference2d(image)

    def tta_inference(self, img):
        pred = self.do_inference(img)
        for flip_idx in self.tta_flips:
            pred += flip(self.do_inference(flip(img, flip_idx)), flip_idx)
        pred /= len(self.tta_flips) + 1
        return pred

    def inference2d(self, image):
        batch_modulo = image.shape[2] % self.args.val_batch_size
        if batch_modulo != 0:
            batch_pad = self.args.val_batch_size - batch_modulo
            image = nn.ConstantPad3d((0, 0, 0, 0, batch_pad, 0), 0)(image)
            mark_step(self.args.run_lazy_mode)
        image = torch.transpose(image.squeeze(0), 0, 1)
        preds_shape = (image.shape[0], self.n_class + 1, *image.shape[2:])
        preds = torch.zeros(preds_shape, dtype=image.dtype, device=image.device)
        for start in range(0, image.shape[0] - self.args.val_batch_size + 1, self.args.val_batch_size):
            end = start + self.args.val_batch_size
            pred = self.model(image[start:end])
            preds[start:end] = pred.data
            mark_step(self.args.run_lazy_mode)
        if batch_modulo != 0:
            preds = preds[batch_pad:]
            mark_step(self.args.run_lazy_mode)
        return torch.transpose(preds, 0, 1).unsqueeze(0)

    def inference2d_test(self, image):
        preds_shape = (image.shape[0], self.n_class + 1, *image.shape[2:])
        preds = torch.zeros(preds_shape, dtype=image.dtype, device=image.device)
        for depth in range(image.shape[2]):
            preds[:, :, depth] = self.sliding_window_inference(image[:, :, depth])
        return preds

    def sliding_window_inference(self, image):
        if self.args.hpus:
            from models.monai_sliding_window_inference import sliding_window_inference
        else:
            from monai.inferers import sliding_window_inference
        if self.args.run_lazy_mode:
            def predictor(data, *args, **kwargs):
                if self.args.dim == 2 and self.args.exec_mode == "predict":
                    data = data.to('hpu')
                mark_step(self.args.run_lazy_mode)
                out = self.model(data, *args, **kwargs)
                mark_step(self.args.run_lazy_mode)
                return out
            self.predictor = predictor
        else:
            self.predictor = self.model
        return sliding_window_inference(
            inputs=image,
            roi_size=self.patch_size,
            sw_batch_size=self.args.val_batch_size,
            predictor=self.predictor,
            overlap=self.args.overlap,
            mode=self.args.blend,
        )

    def on_validation_epoch_end(self):
        if os.getenv('framework') == 'NPT': #Pytorch and PTL compatibility
             self.current_epoch = self.trainer.current_epoch

        if self.current_epoch < self.args.skip_first_n_eval:
            self.log("dice_sum", 0.001 * self.current_epoch)
            self.dice.reset()
            return None
        if os.getenv('framework') == 'PTL': #Pytorch and PTL compatibility
            loss = torch.stack(self.validation_step_outputs).mean() if not self.current_epoch % 2 else torch.tensor(0)
        else:
            loss = torch.stack(self.validation_step_outputs).mean()
        self.log("val_loss", loss)
        self.validation_step_outputs.clear()
        dice = self.dice.compute()
        dice_sum = torch.sum(dice)
        if dice_sum >= self.best_sum:
            self.best_sum = dice_sum
            self.best_sum_dice = dice[:]
            self.best_sum_epoch = self.current_epoch
        for i, dice_i in enumerate(dice):
            if dice_i > self.best_dice[i]:
                self.best_dice[i], self.best_epoch[i] = dice_i, self.current_epoch

        if is_main_process():
            metrics = {}
            metrics.update({"mean dice": round(torch.mean(dice).item(), 2)})
            metrics.update({"TOP_mean": round(torch.mean(self.best_sum_dice).item(), 2)})
            if self.n_class > 1:
                metrics.update({f"L{i+1}": round(m.item(), 2) for i, m in enumerate(dice)})
                metrics.update({f"TOP_L{i+1}": round(m.item(), 2) for i, m in enumerate(self.best_sum_dice)})
            metrics.update({"val_loss": round(loss.item(), 4)})
            if not self.current_epoch % 2:
                self.dllogger.log(step=self.current_epoch, data=metrics)
                self.dllogger.flush()

        if not self.current_epoch % 2 and os.getenv('framework') == 'PTL': #PyTorch and PTL compatibility
            self.log("val_loss", loss)
            self.log("dice_sum", dice_sum)

        if os.getenv('framework') == 'NPT':  #PyTorch and PTL compatibility
           return {"val_loss":loss, "dice_sum":dice_sum}


    def on_test_epoch_end(self):
        if self.args.exec_mode == "evaluate":
            self.eval_dice = self.dice.compute()

    def on_train_epoch_end(self):
        if not self.args.habana_loader:
            #WA for odd epoch getting skipped
            for dl in self.trainer.train_dataloader:
                pass

    def configure_optimizers(self):
        if self.args.hpus:
            self.model = self.model.to(get_device(self.args))
            if self.args.hpus > 1 and os.environ['framework']=="NPT":
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, bucket_cap_mb=self.args.bucket_cap_mb,
                                                                       gradient_as_bucket_view=True, static_graph=True)
        # Avoid instantiate optimizers if not have to
        # since might not be supported
        if self.args.optimizer.lower() == 'sgd':
            optimizer = SGD(self.parameters(), lr=self.learning_rate, momentum=self.args.momentum)
        elif self.args.optimizer.lower() == 'adam':
            optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay)
        elif self.args.optimizer.lower() == 'radam':
            optimizer = RAdam(self.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay)
        elif self.args.optimizer.lower() == 'adamw':
            optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay)
        elif self.args.optimizer.lower() == 'fusedadamw':
            from habana_frameworks.torch.hpex.optimizers import FusedAdamW
            optimizer = FusedAdamW(self.parameters(), lr=self.learning_rate, eps=1e-08, weight_decay=self.args.weight_decay)
        else:
            assert False, "optimizer {} not suppoerted".format(self.args.optimizer.lower())

        scheduler = {
            "none": None,
            "multistep": torch.optim.lr_scheduler.MultiStepLR(optimizer, self.args.steps, gamma=self.args.factor),
            "cosine": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_epochs),
            "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=self.args.factor, patience=self.args.lr_patience, verbose=True
            ),
        }[self.args.scheduler.lower()]

        opt_dict = {"optimizer": optimizer, "monitor": "val_loss"}
        if scheduler is not None:
            opt_dict.update({"lr_scheduler": scheduler})
        return opt_dict

    def save_mask(self, pred):
        if self.test_idx == 0:
            data_path = get_path(self.args)
            self.test_imgs, _ = get_test_fnames(self.args, data_path)
        fname = os.path.basename(self.test_imgs[self.test_idx]).replace("_x", "")
        np.save(os.path.join(self.save_dir, fname), pred, allow_pickle=False)
        self.test_idx += 1

    def get_train_data(self, batch):
        img, lbl = batch["image"], batch["label"]
        if self.args.dim == 2 and self.args.data2d_dim == 3:
            img, lbl = layout_2d(img, lbl)
        if self.args.hpus:
            img, lbl = img.to(torch.device("hpu"), non_blocking=False), lbl.to(torch.device("hpu"), non_blocking=False)
        return img, lbl
    
    
class LiteHRNet(NNUnet):
    def __init__(self, args):
        super(NNUnet, self).__init__()
        self.validation_step_outputs = []
        self.args = args
        self.n_class = 19 if self.args.dataset == "kitti" else 4
        if not hasattr(self.args, "drop_block"):  # For backward compability
            self.args.drop_block = False
        if hasattr(self, 'save_hyperparameters'): #Pytorch and PTL compatibility
            self.save_hyperparameters()
        self.build_nnunet(self.args.dataset)
        self.loss = nn.CrossEntropyLoss()
        self.dice = Dice(self.n_class)
        if hasattr(self.args, 'use_torch_compile') and self.args.use_torch_compile:
            self.loss = torch.compile(self.loss, backend="aot_hpu_training_backend")
            self.dice = torch.compile(self.dice, backend="aot_hpu_training_backend")
        self.first = True
        self.best_sum = 0
        self.best_sum_epoch = 0
        self.best_dice = self.n_class * [0]
        self.best_epoch = self.n_class * [0]
        self.best_sum_dice = self.n_class * [0]
        self.learning_rate = args.learning_rate
        self.tta_flips = get_tta_flips(args.dim)
        self.test_idx = 0
        self.test_imgs = []
        if self.args.exec_mode in ["train", "evaluate"]:
            self.dllogger = get_dllogger(args.results)
        self.window_size = 20
        self.train_loss_valid_end = 0
        self.train_loss = torch.zeros(self.window_size, device=torch.device(get_device_str(self.args)))
        
    def build_nnunet(self, dataset="brats"):
        class _LiteHRNet(nn.Module):
            img_meta = {
                "kitti": [{
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
                "brats": [{
                    'ori_shape': (192, 160, 3),
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
                }]
            }
            def __init__(self, segmentor, dataset):
                super().__init__()
                self.segmentor = segmentor
                self.dataset = dataset
                
            def forward(self, x):
                rescale = not self.training
                if not self.training and self.dataset == "kitti":
                    x = x.transpose(2, 3).transpose(1, 2)
                    x -= torch.tensor([[123.675, 116.28, 103.53]], dtype=x.dtype, device=x.device)
                    x /= torch.tensor([[58.395, 57.12, 57.375]], dtype=x.dtype, device=x.device)
                return self.segmentor.whole_inference(
                    img=x,
                    img_meta=self.img_meta[self.dataset],
                    rescale=rescale)
        
        if dataset == "brats":
            config[self.args.model]["decode_head"]["num_classes"] = self.n_class
        segmentor = build_segmentor(config[self.args.model])
        self.model = _LiteHRNet(segmentor, dataset)
        if hasattr(self.args, 'use_torch_compile') and self.args.use_torch_compile:
            self.model = torch.compile(self.model, backend="aot_hpu_training_backend")
            
    def training_step(self, batch, batch_idx):
        with torch.autocast(device_type=get_device_str(self.args), dtype=get_device_data_type(self.args), enabled=self.args.is_autocast):
            img, lbl = self.get_train_data(batch)
            if self.args.dataset == "brats": ###
                img = img[:,:3]
                lbl = lbl[:,0].long()
            pred = self.model(img)
            loss = self.compute_loss(pred, lbl)
        # WA for https://github.com/Lightning-AI/lightning/issues/17251
        # TBD: move to use of trochmetrics==v1.0.0 for following calc
        if batch_idx  % self.args.progress_bar_refresh_rate == 0:
            self.train_loss = torch.cat((torch.tensor([loss], device=self.train_loss.device), self.train_loss[:-1]))
            if self.train_loss_valid_end < self.window_size:
                self.train_loss_valid_end += 1
                mask = (torch.arange(len(self.train_loss)) >= 0) & (torch.arange(len(self.train_loss)) <= self.train_loss_valid_end)
                moving_average_loss = torch.mean(self.train_loss[mask])
            else:
                moving_average_loss = torch.mean(self.train_loss)
            self.log("loss", moving_average_loss, prog_bar=True)
        return loss
            
    def validation_step(self, batch, batch_idx):
        with torch.autocast(device_type=get_device_str(self.args), dtype=get_device_data_type(self.args), enabled=self.args.is_autocast):
            if os.getenv('framework') == 'NPT': #Pytorch and PTL compatibility
                self.current_epoch = self.trainer.current_epoch
            if self.current_epoch < self.args.skip_first_n_eval:
                return None
            img, lbl = batch["image"], batch["label"]
            if self.args.hpus:
                img, lbl = img.to(torch.device("hpu"), non_blocking=False), lbl.to(torch.device("hpu"), non_blocking=False)
            if self.args.dataset == "brats": ###
                img = img[:,:3]
                lbl = lbl[:,0].long()
            elif self.args.dataset == "kitti":
                img, lbl = img.to(torch.float32), lbl.to(torch.int32)
            pred = self.forward(img)
            if self.args.dim == 3:
                mark_step(self.args.run_lazy_mode)
            #Calculating dice update before calculating loss as dice update has blocking calls of "if conditions" to fetch tensors to CPU
            if self.args.dataset == "brats": ###
                self.dice.update(pred, lbl[:, 0])
            elif self.args.dataset == "kitti":
                self.dice.update(pred, lbl[:, ..., 0])
            loss = self.loss(pred, lbl)
            mark_step(self.args.run_lazy_mode)
            self.validation_step_outputs.append(loss)
            return {"val_loss": loss}
        
    def inference2d(self, image):
        batch_modulo = image.shape[2] % self.args.val_batch_size
        if batch_modulo != 0:
            batch_pad = self.args.val_batch_size - batch_modulo
            image = nn.ConstantPad3d((0, 0, 0, 0, batch_pad, 0), 0)(image)
            mark_step(self.args.run_lazy_mode)
        image = torch.transpose(image.squeeze(0), 0, 1)
        preds_shape = (image.shape[0], self.n_class, *image.shape[2:])
        preds = torch.zeros(preds_shape, dtype=image.dtype, device=image.device)
        for start in range(0, image.shape[0] - self.args.val_batch_size + 1, self.args.val_batch_size):
            end = start + self.args.val_batch_size
            pred = self.model(image[start:end])
            preds[start:end] = pred.data
            mark_step(self.args.run_lazy_mode)
        if batch_modulo != 0:
            preds = preds[batch_pad:]
            mark_step(self.args.run_lazy_mode)
        return torch.transpose(preds, 0, 1).unsqueeze(0)
