"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.cifaredge_model import CifarEdgeModel
from util.visualizer import Visualizer
from util import html

import torch
import torchvision.transforms as trn
import torchvision.datasets as dset
from adv import option, pgd, wrn

class PrivateOptions(option.BaseOptions):
    def initialize(self):
        option.BaseOptions.initialize(self)

        # WRN Architecture
        self.parser.add_argument('--layers', default=28, type=int, help='total number of layers')
        self.parser.add_argument('--widen-factor', default=10, type=int, help='widen factor')
        self.parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability')

arg = PrivateOptions().parse()

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(
                                         mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(
                                         mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

if arg.dataset == 'cifar10':
    train_data = dset.CIFAR10(arg.dataroot, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR10(arg.dataroot, train=False, transform=test_transform)
    num_classes = 10
else:
    train_data = dset.CIFAR100(arg.dataroot, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR100(arg.dataroot, train=False, transform=test_transform)
    num_classes = 100

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=arg.batch_size, shuffle=True,
    num_workers=arg.prefetch, pin_memory=torch.cuda.is_available())
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=arg.test_bs, shuffle=False,
    num_workers=arg.prefetch, pin_memory=torch.cuda.is_available())

# Create model
if arg.model == 'wrn':
    net = wrn.WideResNet(arg.layers, num_classes, arg.widen_factor, dropRate=arg.droprate)
else:
    assert False, arg.model + ' is not supported.'

adversary_test = pgd.PGD(epsilon=arg.epsilon * 2, num_steps=arg.test_num_steps, step_size=arg.test_step_size * 2).cuda()

opt = TestOptions().parse()

if len(opt.gpu_ids) > 0:
    net = torch.nn.DataParallel(net, device_ids=opt.gpu_ids)
    net.cuda()
    torch.cuda.manual_seed(arg.random_seed)

dataloader = data.create_dataloader(opt)

model = CifarEdgeModel(opt)
model.eval()

# visualizer = Visualizer(opt)
#
# # create a webpage that summarizes the all results
# web_dir = os.path.join(opt.results_dir, opt.name,
#                        '%s_%s' % (opt.phase, opt.which_epoch))
# webpage = html.HTML(web_dir,
#                     'Experiment = %s, Phase = %s, Epoch = %s' %
#                     (opt.name, opt.phase, opt.which_epoch))
#
# # test
# for i, data_i in enumerate(dataloader):
#     if i * opt.batchSize >= opt.how_many:
#         break
#
#     generated = model(data_i, mode='inference')
#
#     img_path = data_i['path']
#     for b in range(generated.shape[0]):
#         print('process image... %s' % img_path[b])
#         visuals = OrderedDict([('input_label', data_i['label'][b]),
#                                ('synthesized_image', generated[b])])
#         visualizer.save_images(webpage, visuals, img_path[b:b + 1])
#
# webpage.save()

# robustness test

net.eval()
loss_avg = 0.0
correct = 0
adv_loss_avg = 0.0
adv_correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data_i = {}
        data_i['label'] = target
        data_i['instance'] = torch.zeros(data.shape[0])
        data_i['image'] = data
        generated = model(data_i, mode='inference')

        data, target = data.cuda(), target.cuda()

        adv_data = adversary_test(net, data, target)

        # forward
        output = net(data)
        loss = F.cross_entropy(output, target)

        # accuracy
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).sum().item()

        # test loss average
        loss_avg += float(loss.data)

        # forward
        adv_output = net(adv_data)
        adv_loss = F.cross_entropy(adv_output, target)

        # accuracy
        adv_pred = adv_output.data.max(1)[1]
        adv_correct += adv_pred.eq(target.data).sum().item()

        # test loss average
        adv_loss_avg += float(adv_loss.data)

test_loss = loss_avg / len(test_loader)
test_accuracy = correct / len(test_loader.dataset)
adv_test_loss = adv_loss_avg / len(test_loader)
adv_test_accuracy = adv_correct / len(test_loader.dataset)

print('test_loss:', test_loss)
print('test_accuracy:', test_accuracy)
print('adv_test_loss:', adv_test_loss)
print('adv_test_accuracy:', adv_test_accuracy)






