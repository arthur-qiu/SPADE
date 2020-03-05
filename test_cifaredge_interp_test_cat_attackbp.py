"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.cifaredge_interp_model import CifarInterpEdgeModel
from util.visualizer import Visualizer
from util import html

import torch
import torch.nn as nn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from adv import pgd, wrn
from torch import optim

class InterpNets(nn.Module):
    def __init__(self, net1, net2, mark1 = None, mark2 = None):
        super(InterpNets, self).__init__()
        self.net1 = net1
        self.net2 = net2
        self.mark1 = mark1
        self.mark2 = mark2
        self.iters_interp = 10

    def forward(self, x):

        generated1 = self.net1(x, self.mark1)
        generated2 = self.net1(x, self.mark2)

        interp_z = torch.zeros_like(x).uniform_(0, 1).cuda()

        interp_z.requires_grad = True
        interp_z_min = torch.zeros_like(interp_z).cuda()
        interp_z_max = torch.zeros_like(interp_z).cuda() + 1
        attack_optimizer = optim.Adam([interp_z], lr=0.01)
        for i in range(self.iters_interp):
            interp_generated = torch.min(torch.max(interp_z, interp_z_min), interp_z_max) * generated1 + (
                        1 - torch.min(torch.max(interp_z, interp_z_min), interp_z_max)) * generated2
            interp_loss = criterionL2(interp_generated, x)
            attack_optimizer.zero_grad()
            interp_loss.backward()
            attack_optimizer.step()

        interp_z = interp_z.clamp(0, 1)

        generated = interp_z * generated1 + (1 - interp_z) * generated2

        return self.net2(generated)

opt = TestOptions().parse()

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(
                                         mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(
                                         mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

if opt.dataset == 'cifar10':
    train_data = dset.CIFAR10(opt.dataroot, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR10(opt.dataroot, train=False, transform=test_transform)
    num_classes = 10
else:
    train_data = dset.CIFAR100(opt.dataroot, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR100(opt.dataroot, train=False, transform=test_transform)
    num_classes = 100

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=opt.batch_size, shuffle=True,
    num_workers=opt.prefetch, pin_memory=torch.cuda.is_available())
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=opt.test_bs, shuffle=False,
    num_workers=opt.prefetch, pin_memory=torch.cuda.is_available())

# Create model
if opt.cls_model == 'wrn':
    net = wrn.WideResNet(opt.layers, num_classes, opt.widen_factor, dropRate=opt.droprate)
else:
    assert False, opt.model + ' is not supported.'

if len(opt.gpu_ids) > 0:
    net = torch.nn.DataParallel(net, device_ids=opt.gpu_ids)
    net.cuda()
    torch.cuda.manual_seed(opt.random_seed)

start_epoch = opt.start_epoch
# Restore model if desired
if opt.load != '':
    if opt.test and os.path.isfile(opt.load):
        net.load_state_dict(torch.load(opt.load))
        print('Appointed Model Restored!')
    else:
        model_name = os.path.join(opt.load, opt.dataset + opt.model +
                                  '_epoch_' + str(start_epoch) + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', start_epoch)
        else:
            raise Exception("Could not resume")

adversary_test = pgd.PGD(epsilon=opt.epsilon * 2, num_steps=opt.test_num_steps, step_size=opt.test_step_size * 2).cuda()

dataloader = data.create_dataloader(opt)

model = CifarInterpEdgeModel(opt)
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
iters_interp = 10
criterionL2 = torch.nn.MSELoss()
for data, target in test_loader:

    data, target = data.cuda(), target.cuda()

    interp_z = torch.zeros_like(data).uniform_(0, 1).cuda()

    generated1 = model(data, mode='just_fw1').detach().cuda()

    generated2 = model(data, mode='just_cat2').detach().cuda()

    interp_z.requires_grad = True
    interp_z_min = torch.zeros_like(interp_z).cuda()
    interp_z_max = torch.zeros_like(interp_z).cuda() + 1
    optimizer = optim.Adam([interp_z], lr=0.01)
    for i in range(iters_interp):
        interp_generated = torch.min(torch.max(interp_z,interp_z_min),interp_z_max) * generated1 + (1 - torch.min(torch.max(interp_z,interp_z_min),interp_z_max)) * generated2
        interp_loss = criterionL2(interp_generated, data)
        optimizer.zero_grad()
        interp_loss.backward()
        optimizer.step()

    interp_z = interp_z.clamp(0,1)
    interp_generated = interp_z * generated1 + (1 - interp_z) * generated2

    attack_interp_z = torch.zeros_like(data).uniform_(0, 1).cuda()
    two_nets = InterpNets(model, net, attack_interp_z, 'just_fw1', 'just_cat2')

    adv_data = adversary_test(two_nets, data, target)

    # forward
    output = net(interp_generated)
    loss = F.cross_entropy(output, target)

    # accuracy
    pred = output.data.max(1)[1]
    correct += pred.eq(target.data).sum().item()

    # test loss average
    loss_avg += float(loss.data)

    # forward
    adv_interp_z = torch.zeros_like(data).uniform_(0, 1).cuda()

    adv_generated1 = model(adv_data, mode='just_fw1').detach().cuda()

    adv_generated2 = model(adv_data, mode='just_cat2').detach().cuda()

    adv_interp_z.requires_grad = True
    adv_interp_z_min = torch.zeros_like(adv_interp_z).cuda()
    adv_interp_z_max = torch.zeros_like(adv_interp_z).cuda() + 1
    adv_optimizer = optim.Adam([adv_interp_z], lr=0.01)
    for i in range(iters_interp):
        adv_interp_generated = torch.min(torch.max(adv_interp_z,adv_interp_z_min),adv_interp_z_max) * adv_generated1 + (1 - torch.min(torch.max(adv_interp_z,adv_interp_z_min),adv_interp_z_max)) * adv_generated2
        adv_interp_loss = criterionL2(adv_interp_generated, adv_data)
        adv_optimizer.zero_grad()
        adv_interp_loss.backward()
        adv_optimizer.step()

    adv_interp_z = adv_interp_z.clamp(0,1)

    adv_generated = adv_interp_z * adv_generated1 + (1 - adv_interp_z) * adv_generated2

    adv_output = net(adv_generated)
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






