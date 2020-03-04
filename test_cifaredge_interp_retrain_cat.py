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
import json
import time

class TwoNets(nn.Module):
    def __init__(self, net1, net2):
        super(TwoNets, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x):
        return self.net2(self.net1(x))

def train():
    net.train()  # enter train mode
    loss_avg = 0.0
    for bx, by in train_loader:

        bx, by = bx.cuda(), by.cuda()

        interp_z = torch.zeros_like(bx).uniform_(0, 1).cuda()

        generated1 = model(bx, mode='just_fw1').detach().cuda()

        generated2 = model(bx, mode='just_cat2').detach().cuda()

        generated = interp_z * generated1 + (1 - interp_z) * generated2

        # forward
        logits = net(generated)

        # backward
        # scheduler.step()
        optimizer.zero_grad()
        loss = F.cross_entropy(logits, by)
        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    state['train_loss'] = loss_avg

# test function
def test():
    net.eval()
    # two_nets = TwoNets(model, net)
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.cuda(), target.cuda()

            interp_z = torch.zeros_like(data).uniform_(0, 1).cuda()

            generated1 = model(data, mode='just_fw1').detach().cuda()

            generated2 = model(data, mode='just_cat2').detach().cuda()

            generated = interp_z * generated1 + (1 - interp_z) * generated2

            output = net(generated)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)

opt = TestOptions().parse()
state = {k: v for k, v in opt._get_kwargs()}

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
    assert False, opt.cls_model + ' is not supported.'

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
        model_name = os.path.join(opt.load, opt.dataset + opt.cls_model +
                                  '_epoch_' + str(start_epoch) + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', start_epoch)
        else:
            raise Exception("Could not resume")

epoch_step = json.loads(opt.epoch_step)
lr = state['learning_rate']
optimizer = torch.optim.SGD(
    net.parameters(), lr, momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)

adversary_train = pgd.PGD(epsilon=opt.epsilon * 2, num_steps=opt.num_steps, step_size=opt.step_size * 2).cuda()
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

# Make save directory
if not os.path.exists(opt.save):
    os.makedirs(opt.save)
if not os.path.isdir(opt.save):
    raise Exception('%s is not a dir' % opt.save)

with open(os.path.join(opt.save, "log_" + opt.dataset + opt.cls_model +
                                  '_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_accuracy(%)\n')

print('Beginning Training\n')

# Main loop
best_test_accuracy = 0
for epoch in range(start_epoch, opt.epochs + 1):
    state['epoch'] = epoch

    begin_epoch = time.time()

    train()
    test()

    # Save model
    if epoch > 10 and epoch % 10 == 0:
        torch.save(net.state_dict(),
                   os.path.join(opt.save, opt.dataset + opt.cls_model +
                                '_epoch_' + str(epoch) + '.pt'))

    if state['test_accuracy'] > best_test_accuracy:
        best_test_accuracy = state['test_accuracy']
        torch.save(net.state_dict(),
                   os.path.join(opt.save, opt.dataset + opt.cls_model +
                                '_epoch_best.pt'))

    # Show results
    with open(os.path.join(opt.save, "log_" + opt.dataset + opt.cls_model +
                                      '_training_results.csv'), 'a') as f:
        f.write('%03d,%0.6f,%05d,%0.3f,%0.3f,%0.2f\n' % (
            (epoch),
            lr,
            time.time() - begin_epoch,
            state['train_loss'],
            state['test_loss'],
            100. * state['test_accuracy'],
        ))

    print('Epoch {0:3d} | LR {1:.6f} | Time {2:5d} | Train Loss {3:.3f} | Test Loss {4:.3f} | Test Acc {5:.2f}'.format(
        (epoch),
        lr,
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100. * state['test_accuracy'])
    )

    # Adjust learning rate
    if epoch in epoch_step:
        lr = optimizer.param_groups[0]['lr'] * opt.lr_decay_ratio
        optimizer = torch.optim.SGD(
            net.parameters(), lr, momentum=state['momentum'],
            weight_decay=state['decay'], nesterov=True)
        print("new lr:", lr)






