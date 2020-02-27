"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import models.networks as networks
import util.util as util
from util import forward_canny, backward_canny
import math
import torch.optim as optim
import os

class IdentityMapping(nn.Module):
    def __init__(self, base):
        super(IdentityMapping, self).__init__()
        self.module = base
    def forward(self, x):
        x = self.module(x)
        return x

class CifarEdgeModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        self.canny_net = backward_canny.Canny_Net(opt.sigma, opt.high_threshold, opt.low_threshold, opt.robust_threshold)
        if self.use_gpu():
            self.canny_net.cuda()

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

        if opt.cls:
            from adv import pgd, wrn

            # Create model
            if opt.cls_model == 'wrn':
                self.net = wrn.WideResNet(opt.layers, 10, opt.widen_factor, dropRate=opt.droprate)
            else:
                assert False, opt.cls_model + ' is not supported.'

            if len(opt.gpu_ids) > 0:
                assert (torch.cuda.is_available())
                self.net.cuda()

            # Restore model if desired
            if opt.load != '':
                self.net = IdentityMapping(self.net)
                if os.path.isfile(opt.load):
                    self.net.load_state_dict(torch.load(opt.load))
                    print('Appointed Model Restored!')
                else:
                    model_name = os.path.join(opt.load, opt.dataset + opt.cls_model +
                                              '_epoch_' + str(opt.start_epoch) + '.pt')
                    if os.path.isfile(model_name):
                        self.net.load_state_dict(torch.load(model_name))
                        print('Model restored! Epoch:', opt.start_epoch)
                    else:
                        raise Exception("Could not resume")

        if opt.cnn_edge:
            from adv import zip_wrn
            self.edge_net = zip_wrn.BlurZipNet()
            if len(opt.gpu_ids) > 0:
                assert (torch.cuda.is_available())
                self.edge_net.cuda()

            if opt.load != '':
                self.edge_net = IdentityMapping(self.edge_net)
                if os.path.isfile(opt.load):
                    self.edge_net.load_state_dict(torch.load(opt.load))
                    print('Appointed Model Restored!')
                else:
                    model_name = os.path.join(opt.load, opt.dataset + opt.cls_model +
                                              '_epoch_' + str(opt.start_epoch) + '.pt')
                    if os.path.isfile(model_name):
                        self.edge_net.load_state_dict(torch.load(model_name))
                        print('Model restored! Epoch:', opt.start_epoch)
                    else:
                        raise Exception("Could not resume")


    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode = "edge_back"):
        if mode == "edge_forward":
            real_image = data.cuda()
            edge = forward_canny.get_edge(real_image, self.opt.sigma, self.opt.high_threshold, self.opt.low_threshold,
                                          self.opt.robust_threshold).detach()
            fake_image, _ = self.generate_fake(edge, real_image)
            return fake_image

        elif mode == "edge_back":
            real_image = data.cuda()
            edge = self.canny_net(real_image)
            fake_image, _ = self.generate_fake(edge, real_image)
            return fake_image

        input_semantics, real_image = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image)
            return g_loss, generated
        elif mode == 'generator_comb':
            g_loss, generated = self.compute_generator_loss_comb(
                input_semantics, real_image)
            return g_loss, generated
        elif mode == 'generator_cnnedge':
            input_semantics = self.edge_net(input_semantics)
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image)
            return g_loss, generated
        elif mode == 'generator_cls':
            cls_label = data['class'].cuda()
            g_loss, generated = self.compute_generator_loss_cls(
                input_semantics, real_image, self.net, cls_label)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image)
            return d_loss
        elif mode == 'discriminator_cnnedge':
            input_semantics = self.edge_net(input_semantics)
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image)
            return d_loss
        elif mode == 'discriminator_comb':
            d_loss = self.compute_discriminator_loss_comb(
                input_semantics, real_image)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _ = self.generate_fake(input_semantics, real_image)
            return fake_image
        elif mode == 'bp_defense_z':
            fake_image = self.bp_defense_z(input_semantics, real_image)
            return fake_image
        elif mode == 'bp_defense_eps':
            fake_image = self.bp_defense_eps(input_semantics, real_image)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        if opt.cls:
            self.optimizer_cls = torch.optim.SGD(
                self.net.parameters(), opt.learning_rate, momentum=opt.momentum,
                weight_decay=opt.decay, nesterov=True)
            return optimizer_G, optimizer_D, self.optimizer_cls

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

        if self.opt.cls:
            torch.save(self.net.state_dict(),
                       os.path.join(self.opt.save, self.opt.dataset + self.opt.cls_model +
                                    '_epoch_' + str(epoch) + '.pt'))

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)

        return netG, netD, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['instance'] = data['instance'].cuda()
            data['image'] = data['image'].cuda()

        # create one-hot label map
        label_map = data['label']
        if label_map.shape[1] > 1:
            bs, _, h, w = label_map.size()
            nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
                else self.opt.label_nc
            input_label = self.FloatTensor(bs, nc, h, w).zero_()
            input_semantics = input_label.scatter_(1, label_map, 1.0)
        else:
            input_semantics = label_map.float()

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        return input_semantics, data['image']

    def compute_generator_loss(self, input_semantics, real_image):
        G_losses = {}

        fake_image, KLD_loss = self.generate_fake(
            input_semantics, real_image, compute_kld_loss=self.opt.use_vae)

        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                * self.opt.lambda_vgg

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image, _ = self.generate_fake(input_semantics, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

    def compute_discriminator_loss_comb(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image, _ = self.generate_fake_comb(input_semantics, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        fake_image = self.netG(input_semantics, z=z)

        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, KLD_loss

    def generate_fake_comb(self, input_semantics, real_image, compute_kld_loss=False):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        fake_image = self.netG(input_semantics, real_image, z=z)

        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, KLD_loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def compute_generator_loss_cls(self, input_semantics, real_image, cls_model, cls_label):
        G_losses = {}

        fake_image, KLD_loss = self.generate_fake(
            input_semantics, real_image, compute_kld_loss=self.opt.use_vae)

        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False)

        G_losses['CLS'] = torch.nn.functional.cross_entropy(cls_model(fake_image),cls_label) * 10

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                * self.opt.lambda_vgg

        return G_losses, fake_image

    def compute_generator_loss_comb(self, input_semantics, real_image):
        G_losses = {}

        fake_image, KLD_loss = self.generate_fake_comb(
            input_semantics, real_image, compute_kld_loss=self.opt.use_vae)

        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                * self.opt.lambda_vgg

        return G_losses, fake_image




    def fw_defense_z(self,input_semantics, z):
        fake_image = self.netG(input_semantics, z=z)
        return fake_image

    def fw_defense_eps(self,input_semantics, real_image, eps):
        mu, logvar = self.netE(real_image)
        std = torch.exp(0.5 * logvar)
        z = eps.mul(std) + mu
        fake_image = self.netG(input_semantics, z=z)
        return fake_image

    def adjust_lr(self, optimizer, cur_lr, decay_rate=0.1, global_step=1, rec_iter=200):

        lr = cur_lr * decay_rate ** (global_step / int(math.ceil(rec_iter * 0.8)))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def bp_defense_z(self, input_semantics, real_image, lr = 0.01, rec_iter = 20, rec_restart = 1, input_latent = 32):

        # the output of R random different initializations of z from L steps of GD
        z_hats_recs = torch.Tensor(rec_restart, real_image.shape[0], input_latent)

        # the R random differernt initializations of z before L steps of GD
        z_hats_orig = torch.Tensor(rec_restart, real_image.shape[0], input_latent)

        for idx in range(rec_restart):
            z_hat = torch.randn(real_image.shape[0], input_latent).cuda()

            z_hat = z_hat.detach().requires_grad_()
            # input_semantics = input_semantics.detach().requires_grad_()

            cur_lr = lr

            optimizer = optim.SGD([z_hat], lr=cur_lr, momentum=0.7)

            z_hats_orig[idx] = z_hat.clone().cpu().detach()

            loss = nn.MSELoss()

            for iteration in range(rec_iter):

                with torch.enable_grad():

                    optimizer.zero_grad()

                    fake_image = self.fw_defense_z(input_semantics, z_hat)

                    reconstruct_loss = loss(fake_image, real_image)

                    reconstruct_loss.backward()

                    optimizer.step()

                cur_lr = self.adjust_lr(optimizer, cur_lr, global_step=3, rec_iter=rec_iter)

            z_hats_recs[idx] = z_hat.cpu().detach().clone()

        reconstructions = torch.Tensor(rec_restart)

        z_hats_recs = z_hats_recs.cuda()

        for i in range(rec_restart):

            fake_image = self.fw_defense_z(input_semantics, z_hats_recs[i])

            reconstructions[i] = loss(fake_image, real_image).cpu().item()

        min_idx = torch.argmin(reconstructions)

        final_out = self.fw_defense_z(input_semantics, z_hats_recs[min_idx])

        return final_out



    def bp_defense_eps(self, input_semantics, real_image, lr = 0.01, rec_iter = 20, rec_restart = 1, input_latent = 32):

        # the output of R random different initializations of z from L steps of GD
        z_hats_recs = torch.Tensor(rec_restart, real_image.shape[0], input_latent)

        # the R random differernt initializations of z before L steps of GD
        z_hats_orig = torch.Tensor(rec_restart, real_image.shape[0], input_latent)

        for idx in range(rec_restart):
            z_hat = torch.randn(real_image.shape[0], input_latent).cuda()

            z_hat = z_hat.detach().requires_grad_()
            # input_semantics = input_semantics.detach().requires_grad_()

            cur_lr = lr

            optimizer = optim.SGD([z_hat], lr=cur_lr, momentum=0.7)

            z_hats_orig[idx] = z_hat.clone().cpu().detach()

            loss = nn.MSELoss()

            for iteration in range(rec_iter):

                with torch.enable_grad():

                    optimizer.zero_grad()

                    fake_image = self.fw_defense_eps(input_semantics, real_image, z_hat)

                    reconstruct_loss = loss(fake_image, real_image)

                    reconstruct_loss.backward()

                    optimizer.step()

                cur_lr = self.adjust_lr(optimizer, cur_lr, global_step=3, rec_iter=rec_iter)

            z_hats_recs[idx] = z_hat.cpu().detach().clone()

        reconstructions = torch.Tensor(rec_restart)

        z_hats_recs = z_hats_recs.cuda()

        for i in range(rec_restart):

            fake_image = self.fw_defense_eps(input_semantics, real_image, z_hats_recs[i])

            reconstructions[i] = loss(fake_image, real_image).cpu().item()

        min_idx = torch.argmin(reconstructions)

        final_out = self.fw_defense_eps(input_semantics, real_image, z_hats_recs[min_idx])

        return final_out


    #TODO finish