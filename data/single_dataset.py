"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
from util import forward_canny
import os


class SingleDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):

        # OPT for GAN
        parser.add_argument('--sigma', type=float, default=1.0)
        parser.add_argument('--high_threshold', type=float, default=0.3)
        parser.add_argument('--low_threshold', type=float, default=0.2)
        parser.add_argument('--robust_threshold', type=float, default=0.3)

        return parser

    def initialize(self, opt):
        self.opt = opt

        image_paths = self.get_paths(opt)

        util.natural_sort(image_paths)

        image_paths = image_paths[:opt.max_dataset_size]

        self.image_paths = image_paths

        size = len(self.image_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        image_paths = []
        assert False, "A subclass of SingleDataset must override self.get_paths(self, opt)"
        return image_paths

    def __getitem__(self, index):
        # input image (real images)
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = image.convert('RGB')

        params = get_params(self.opt, image.size)

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        label_tensor = forward_canny.get_edge(image_tensor.unsqueeze(0), self.opt.sigma, self.opt.high_threshold, self.opt.low_threshold, self.opt.robust_threshold)

        input_dict = {'label': label_tensor[0],
                      'image': image_tensor,
                      'path': image_path,
                      'instance': 0,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size
