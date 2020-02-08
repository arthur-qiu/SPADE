"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.single_dataset import SingleDataset
from data.image_folder import make_dataset


class CifarDataset(SingleDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = SingleDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        load_size = 36 if is_train else 32
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=32)
        parser.set_defaults(display_winsize=32)
        parser.set_defaults(label_nc=1)
        parser.set_defaults(contain_dontcare_label=False)

        parser.add_argument('--image_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        return parser

    def get_paths(self, opt):
        image_dir = opt.image_dir
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        return image_paths
