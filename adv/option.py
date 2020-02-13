import argparse
import os

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                            help='Choose between CIFAR-10, CIFAR-100.')
        self.parser.add_argument('--model', '-m', type=str, default='wrn',
                            choices=['wrn'], help='Choose architecture.')
        # Optimization options
        self.parser.add_argument('--epochs', '-e', type=int, default=50, help='Number of epochs to train.')
        self.parser.add_argument('--start_epoch', type=int, default=1, help='The start epoch to train. Design for restart.')
        self.parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
        self.parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
        self.parser.add_argument('--test_bs', type=int, default=128)
        self.parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
        self.parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
        self.parser.add_argument('--epoch_step', default='[40,42,44,46,48]', type=str,
                            help='json list with epochs to drop lr on')
        self.parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
        # Checkpoints
        self.parser.add_argument('--save', '-s', type=str, default='./logs/cifar10_adv', help='Folder to save checkpoints.')
        self.parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
        self.parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
        self.parser.add_argument('--dataroot', default='.', type=str)
        # Acceleration
        self.parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
        self.parser.add_argument('--prefetch', type=int, default=1, help='Pre-fetching threads.')
        # Adversarial setting
        self.parser.add_argument('--epsilon', type=float, default=8 / 255,
                            help='perturbation')
        self.parser.add_argument('--num_steps', type=int, default=7,
                            help='perturb number of steps')
        self.parser.add_argument('--step_size', type=float, default=2 / 255,
                            help='perturb step size')
        self.parser.add_argument('--test_num_steps', type=int, default=20,
                            help='test perturb number of steps')
        self.parser.add_argument('--test_step_size', type=float, default=2 / 255,
                            help='test perturb step size')

        # Others
        self.parser.add_argument('--random_seed', type=int, default=1)

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        # Make save directory
        if not os.path.exists(self.opt.save):
            os.makedirs(self.opt.save)
        if not os.path.isdir(self.opt.save):
            raise Exception('%s is not a dir' % self.opt.save)
        if save and not self.opt.test:
            file_name = os.path.join(self.opt.save, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
