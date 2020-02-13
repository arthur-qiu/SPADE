import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import utils

def normalize_l2(x):
    """
    Expects x.shape == [N, C, H, W]
    """
    norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1)
    norm = norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return x / norm

class CWLoss(nn.Module):
    def __init__(self, num_classes, margin=50, reduce=True):
        super(CWLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.reduce = reduce
        return

    def forward(self, logits, targets):
        """
        :param inputs: predictions
        :param targets: target labels
        :return: loss
        """
        onehot_targets = utils.one_hot_tensor(targets, self.num_classes)

        self_loss = torch.sum(onehot_targets * logits, dim=1)
        # other_loss = torch.max((1 - onehot_targets) * logits, dim=1)[0]
        other_loss = torch.max(
            (1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

        loss = -torch.sum(torch.clamp(self_loss - other_loss + self.margin, 0))

        if self.reduce:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num

        return loss

class PGD(nn.Module):
    def __init__(self, epsilon, num_steps, step_size, data_min = -1.0, data_max = 1.0, grad_sign=True, random_start = True):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign
        self.data_min = data_min
        self.data_max = data_max
        self.random_start = random_start

    def forward(self, model, bx, by):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """
        adv_bx = bx.detach().clone()
        if self.random_start:
            adv_bx += torch.zeros_like(adv_bx).uniform_(-self.epsilon, self.epsilon)
            adv_bx = adv_bx.clamp(self.data_min, self.data_max)

        for i in range(self.num_steps):
            adv_bx.requires_grad_()
            with torch.enable_grad():
                logits = model(adv_bx)
                if len(logits) == 2:
                    logits = logits[1]
                loss = F.cross_entropy(logits, by, reduction='sum')
            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]

            if self.grad_sign:
                adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())
            else:
                grad = normalize_l2(grad.detach())
                adv_bx = adv_bx.detach() + self.step_size * grad

            adv_bx = torch.min(torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(self.data_min, self.data_max)

        return adv_bx


class FGSM(nn.Module):
    def __init__(self, epsilon, data_min = -1.0, data_max = 1.0, grad_sign=True, random_start = True):
        super().__init__()
        self.epsilon = epsilon
        self.step_size = epsilon
        self.grad_sign = grad_sign
        self.data_min = data_min
        self.data_max = data_max
        self.random_start = random_start

    def forward(self, model, bx, by):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """
        adv_bx = bx.detach().clone()
        if self.random_start:
            adv_bx += torch.zeros_like(adv_bx).uniform_(-self.epsilon, self.epsilon)
            adv_bx = adv_bx.clamp(self.data_min, self.data_max)

        adv_bx.requires_grad_()
        with torch.enable_grad():
            logits = model(adv_bx)
            if len(logits) == 2:
                logits = logits[1]
            loss = F.cross_entropy(logits, by, reduction='sum')
        grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]

        if self.grad_sign:
            adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())
        else:
            grad = normalize_l2(grad.detach())
            adv_bx = adv_bx.detach() + self.step_size * grad

        adv_bx = torch.min(torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(self.data_min, self.data_max)

        return adv_bx


class PGD_Target(nn.Module):
    def __init__(self, epsilon, num_steps, step_size, data_min = -1.0, data_max = 1.0, grad_sign=True, random_start = True):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign
        self.data_min = data_min
        self.data_max = data_max
        self.random_start = random_start

    def forward(self, model, bx, by):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """
        adv_bx = bx.detach().clone()
        if self.random_start:
            adv_bx += torch.zeros_like(adv_bx).uniform_(-self.epsilon, self.epsilon)
            adv_bx = adv_bx.clamp(self.data_min, self.data_max)

        for i in range(self.num_steps):
            adv_bx.requires_grad_()
            with torch.enable_grad():
                logits = model(adv_bx)
                if len(logits) == 2:
                    logits = logits[1]
                loss = -F.cross_entropy(logits, by, reduction='sum')
            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]

            if self.grad_sign:
                adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())
            else:
                grad = normalize_l2(grad.detach())
                adv_bx = adv_bx.detach() + self.step_size * grad

            adv_bx = torch.min(torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(self.data_min, self.data_max)

        return adv_bx

class PGD_Margin_Target(nn.Module):
    def __init__(self, epsilon, num_steps, step_size, data_min = -1.0, data_max = 1.0, grad_sign=True, random_start = True):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign
        self.data_min = data_min
        self.data_max = data_max
        self.random_start = random_start

    def forward(self, model, bx, by, target):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """
        adv_bx = bx.detach().clone()
        if self.random_start:
            adv_bx += torch.zeros_like(adv_bx).uniform_(-self.epsilon, self.epsilon)
            adv_bx = adv_bx.clamp(self.data_min, self.data_max)

        for i in range(self.num_steps):
            adv_bx.requires_grad_()
            with torch.enable_grad():
                logits = model(adv_bx)
                if len(logits) == 2:
                    logits = logits[1]
                true_logits = torch.gather(logits, dim=1, index=by.unsqueeze(-1))
                target_logits = torch.gather(logits, dim=1, index=target.unsqueeze(-1))
                loss = (target_logits - true_logits).mean()
            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]

            if self.grad_sign:
                adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())
            else:
                grad = normalize_l2(grad.detach())
                adv_bx = adv_bx.detach() + self.step_size * grad

            adv_bx = torch.min(torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(self.data_min, self.data_max)

        return adv_bx


class PGD_CWLoss(nn.Module):
    def __init__(self, epsilon, num_steps, step_size, num_classes = 10, data_min = -1.0, data_max = 1.0, grad_sign=True, random_start = True):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign
        self.data_min = data_min
        self.data_max = data_max
        self.random_start = random_start
        self.loss = CWLoss(num_classes)

    def forward(self, model, bx, by):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """
        adv_bx = bx.detach().clone()
        if self.random_start:
            adv_bx += torch.zeros_like(adv_bx).uniform_(-self.epsilon, self.epsilon)
            adv_bx = adv_bx.clamp(self.data_min, self.data_max)

        for i in range(self.num_steps):
            adv_bx.requires_grad_()
            with torch.enable_grad():
                logits = model(adv_bx)
                if len(logits) == 2:
                    logits = logits[1]
                loss = self.loss(logits, by)
            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]

            if self.grad_sign:
                adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())
            else:
                grad = normalize_l2(grad.detach())
                adv_bx = adv_bx.detach() + self.step_size * grad

            adv_bx = torch.min(torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(self.data_min, self.data_max)

        return adv_bx