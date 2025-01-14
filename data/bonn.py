import numpy as np
import os
import sys
import time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import PIL
import imageio
from easydict import EasyDict as edict
import json
import pickle
from ipdb import set_trace

from . import base
import camera
from util import log, debug

from im_util.transforms import tfmat_from_quat_and_translation
from nn.predict import predict_fruitiness


class Dataset(base.Dataset):

    def __init__(self, opt, split="train", subset=None, downsample=None):
        self.raw_H, self.raw_W = 480, 640
        super().__init__(opt, split)
        self.root = opt.data.root or "data/bonn"
        self.path = self.root
        # load/parse metadata
        meta_fname = f'{self.path}/traj_z-backwards.txt'
        self.list = []
        with open(meta_fname) as meta_file:
            self.meta = meta_file.readlines()
        for line in self.meta:
            t_s, px, py, pz, qx, qy, qz, qw = [float(w) for w in line.split()]

            t_us = int(t_s * 1e6)
            fpath = f'images/rgb_{t_us}'
            if opt.data.init_poses:
                tfmat = tfmat_from_quat_and_translation(
                    np.array([qx, qy, qz, qw]), np.array([px, py, pz]))
            else:
                tfmat = np.eye(4)

            self.list.append({'file_path': fpath, 'transform_matrix': tfmat, 'time': t_s})
        self.focal = opt.data.focal
        if downsample:
            self.list = self.list[::downsample]
        if subset:
            if split == 'val':
                # Split the list into N+1 even pieces and then select a validation image at each
                # splitting point.
                val_spacing = len(self.list) // (subset + 1)
                val_idxs = (np.arange(1, subset + 1) * val_spacing).tolist()
                self.list = [self.list[idx] for idx in val_idxs]
            else:
                self.list = self.list[:subset]
        # preload dataset
        if opt.data.preload:
            self.images = self.preload_threading(opt, self.get_image)
            self.cameras = self.preload_threading(opt, self.get_camera, data_str="cameras")
            self.times = self.preload_threading(opt, self.get_time)
        self.sampled = torch.zeros(len(self.list), self.raw_H, self.raw_W, dtype=torch.int64)

    def prefetch_all_data(self, opt):
        assert(not opt.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])
        if opt.fruit_nn:
            # Float mask where pixels belonging to a fruit are 1.0, pixels far from fruits are 0.0,
            # and pixels close to fruits (approximately 10 pixels or less) are a gradient between.
            self.all['fruitiness'] = predict_fruitiness(opt.fruit_nn,
                                                        self.all['image']).to(opt.device)
        self.all['loss'] = None

    def get_all_camera_poses(self, opt):
        pose_raw_all = [torch.tensor(f["transform_matrix"], dtype=torch.float32) for f in self.list]
        pose_canon_all = torch.stack([self.parse_raw_camera(opt, p) for p in pose_raw_all], dim=0)
        return pose_canon_all

    def __getitem__(self, idx):
        opt = self.opt
        sample = dict(idx=idx)
        aug = self.generate_augmentation(opt) if self.augment else None
        image = self.images[idx] if opt.data.preload else self.get_image(opt, idx)
        image = self.preprocess_image(opt, image, aug=aug)
        intr, pose = self.cameras[idx] if opt.data.preload else self.get_camera(opt, idx)
        intr, pose = self.preprocess_camera(opt, intr, pose, aug=aug)
        sampled = self.sampled[idx]
        time = self.times[idx] if opt.data.preload else self.get_time(opt, idx)
        sample.update(
            image=image,
            intr=intr,
            pose=pose,
            sampled=sampled,
            time=time
        )
        return sample

    def get_image(self, opt, idx):
        image_fname = "{}/{}.png".format(self.path, self.list[idx]["file_path"])
        # directly using PIL.Image.open() leads to weird corruption....
        image = PIL.Image.fromarray(imageio.imread(image_fname))
        return image

    def get_time(self, opt, idx):
        time = self.list[idx]['time']
        return time

    def preprocess_image(self, opt, image, aug=None):
        image = super().preprocess_image(opt, image, aug=aug)
        rgb, mask = image[:3], image[3:]
        return rgb

    def get_camera(self, opt, idx):
        intr = torch.tensor([[self.focal, 0, self.raw_W/2],
                             [0, self.focal, self.raw_H/2],
                             [0, 0, 1]]).float()
        pose_raw = torch.tensor(self.list[idx]["transform_matrix"], dtype=torch.float32)
        pose = self.parse_raw_camera(opt, pose_raw)
        return intr, pose

    def parse_raw_camera(self, opt, pose_raw):
        # BARF uses poses that represent camera-from-world transforms with the x-y-z axes pointing
        # right-up-backwards, respectively.
        pose = camera.pose.invert(pose_raw[:3])

        return pose
