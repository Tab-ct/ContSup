'''
Greedy Local Learning for Resnet in Pytorch

Reference:
Wang Y, Ni Z, Song S, et al. Revisiting locally supervised learning: an alternative to end-to-end training. In ICLR 2021.

#################################

Last modified by authors of ContSup on May 23rd, 2023

'''

import argparse
import os
import shutil
import time
import errno
import math
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from networks_dev.configs import local_loss_scale, local_loss_scale_for_memory_balance

import networks_dev.resnet


parser = argparse.ArgumentParser(description='ContSup-PyTorch')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset: [cifar10|stl10|svhn]')

parser.add_argument('--model', default='resnet', type=str,
                    help='resnet is supported currently')

parser.add_argument('--layers', default=0, type=int,
                    help='total number of layers (have to be explicitly given!)')

parser.add_argument('--droprate', default=0.0, type=float,
                    help='dropout probability (default: 0.0)')

parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.set_defaults(augment=True)

parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--name', default='', type=str,
                    help='name of experiment')
parser.add_argument('--no', default='1', type=str,
                    help='index of the experiment (for recording convenience)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')


# Cosine learning rate
parser.add_argument('--cos_lr', dest='cos_lr', action='store_true',
                    help='whether to use cosine learning rate')
parser.set_defaults(cos_lr=True)


# ContSup
parser.add_argument('--local_module_num', default=1, type=int,
                    help='number of local modules (1 refers to end-to-end training)')

parser.add_argument('--balanced_memory', default=False, type=bool,
                    help='balanced_memory')


parser.add_argument('--aux_net_config', default='1c2f', type=str,
                    help='architecture of auxiliary classifier / contrastive head '
                         '(default: 1c2f; 0c1f refers to greedy SL)'
                         '[0c1f|0c2f|1c1f|1c2f|1c3f|2c2f]')

parser.add_argument('--local_loss_mode', default='contrast', type=str,
                    help='ways to estimate the task-relevant info I(h, y)'
                         '[contrast|cross_entropy]')

parser.add_argument('--aux_net_widen', default=1.0, type=float,
                    help='widen factor of the two auxiliary nets (default: 1.0)')

parser.add_argument('--aux_net_feature_dim', default=128, type=int,
                    help='number of hidden features in auxiliary classifier / contrastive head '
                         '(default: 128)')

#### ContSup ####
parser.add_argument('--context_mode', default='E', type=str,
                    help='the selection of context in format of *RnE* where E for X and Rn for n-neighbors'
                         'R0|R1|R1E|E|R2|R4|R8|R16 .etc.')

parser.add_argument('--h_reconstruct', default=False, type=bool,
                    help='whether to use decoder to reconstruct')

args = parser.parse_args()


# Configurations adopted for training deep networks.
training_configurations = {
    'resnet': {
        'epochs': 160,
        'batch_size': 1024 if args.dataset in ['cifar10', 'svhn'] else 128,
        'initial_learning_rate': 0.8 if args.dataset in ['cifar10', 'svhn'] else 0.1,
        'changing_lr': [80, 120],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
    }
}

record_path = './' \
              + ('Test*_' if args.balanced_memory else 'Test_10_') \
              + str(args.dataset) \
              + '_' + str(args.model) + str(args.layers) \
              + '_K_' + str(args.local_module_num)  \

record_file = record_path + '/training_process.txt'
accuracy_file = record_path + '/accuracy_epoch.txt'
loss_file = record_path + '/loss_epoch.txt'
check_point = os.path.join(record_path, args.checkpoint)


def count_parameters(model):
    ''' Count number of parameters in model influenced by global loss. '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    global best_prec1
    best_prec1 = 0
    global val_acc
    val_acc = []

    class_num = args.dataset in ['cifar10', 'sl10', 'svhn'] and 10 or 100

    if 'cifar' in args.dataset:
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        kwargs_dataset_train = {'train': True}
        kwargs_dataset_test = {'train': False}
    else:
        normalize = transforms.Normalize(mean=[x / 255 for x in [127.5, 127.5, 127.5]],
                                         std=[x / 255 for x in [127.5, 127.5, 127.5]])
        kwargs_dataset_train = {'split': 'train'}
        kwargs_dataset_test = {'split': 'test'}

    if args.augment:
        if 'cifar' in args.dataset:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                  (4, 4, 4, 4), mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            image_size = 32
        elif 'stl' in args.dataset:
            transform_train = transforms.Compose(
                [transforms.RandomCrop(96, padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 normalize])
            image_size = 96
        elif 'svhn' in args.dataset:
            transform_train = transforms.Compose(
                [transforms.RandomCrop(32, padding=2),
                 transforms.ToTensor(),
                 normalize])
            image_size = 32
        else:
            raise NotImplementedError

    else:
        transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 0, 'pin_memory': False}

    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('./data', download=True, transform=transform_train,
                                                **kwargs_dataset_train),
        batch_size=training_configurations[args.model]['batch_size'], shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('./data', download=True, transform=transform_test,
                                                **kwargs_dataset_test),
        batch_size=training_configurations[args.model]['batch_size'], shuffle=False, **kwargs)

    # create model
    if args.model == 'resnet':
        model = eval('networks_dev.resnet.resnet' + str(args.layers))\
            (local_module_num=args.local_module_num,
             batch_size=training_configurations[args.model]['batch_size'],
             image_size=image_size,
             balanced_memory=args.balanced_memory,
             dataset=args.dataset,
             class_num=class_num,
             wide_list=(16, 16, 32, 64),
             dropout_rate=args.droprate,
             aux_net_config=args.aux_net_config,
             local_loss_mode=args.local_loss_mode,
             aux_net_widen=args.aux_net_widen,
             aux_net_feature_dim=args.aux_net_feature_dim,
             context_mode=args.context_mode,
             h_reconstruct=args.h_reconstruct)

    else:
        raise NotImplementedError

    if not os.path.isdir(check_point):
        mkdir_p(check_point)

    cudnn.benchmark = True

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=training_configurations[args.model]['initial_learning_rate'],
                                momentum=training_configurations[args.model]['momentum'],
                                nesterov=training_configurations[args.model]['nesterov'],
                                weight_decay=training_configurations[args.model]['weight_decay'])

    model = torch.nn.DataParallel(model).cuda()

    # print the parameters of the model
    print('Model {} has {} parameters influenced by global loss'.format(args.model+str(args.layers)+' + '+\
                    'local_module:' + str(args.local_module_num)+'+ context_mode:'+args.context_mode + ': ', count_parameters(model)))

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        val_acc = checkpoint['val_acc']
        best_prec1 = checkpoint['best_acc']
        np.savetxt(accuracy_file, np.array(val_acc))
    else:
        start_epoch = 0

    for epoch in range(start_epoch, training_configurations[args.model]['epochs']):

        adjust_learning_rate(optimizer, epoch + 1)

        # train for one epoch
        train(train_loader, model, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_prec1,
            'optimizer': optimizer.state_dict(),
            'val_acc': val_acc,

        }, is_best, checkpoint=check_point)
        print('Best accuracy: ', best_prec1)
        np.savetxt(accuracy_file, np.array(val_acc))


def train(train_loader, model, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(train_loader)

    # switch to train mode
    model.train()

    ixx_1,ixy_1,ixx_2,ixy_2 = local_loss_scale[str(args.model) + str(args.layers)][str(args.dataset)][args.local_module_num] if not args.balanced_memory else \
        local_loss_scale_for_memory_balance[str(args.model) + str(args.layers)][str(args.dataset)][args.local_module_num]

    end = time.time()
    for i, (x, target) in enumerate(train_loader):
        target = target.cuda()
        x = x.cuda()

        optimizer.zero_grad()
        output, loss = model(img=x,
                             target=target,
                             ixx_1=ixx_1,
                             ixy_1=ixy_1,
                             ixx_2=ixx_2,
                             ixy_2=ixy_2)
        optimizer.step()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), x.size(0))
        top1.update(prec1.item(), x.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:

            string = ('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                       epoch, i+1, train_batches_num, batch_time=batch_time,
                       loss=losses, top1=top1))
            print(string)

def validate(val_loader, model, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(val_loader)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        with torch.no_grad():
            output, loss = model(img=input_var,
                                 target=target_var,)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    fd = open(record_file, 'a+')
    string = ('Test: [{0}][{1}/{2}]\t'
              'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
              'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
              'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
        epoch, (i + 1), train_batches_num, batch_time=batch_time,
        loss=losses, top1=top1))
    print(string)

    fd.write(string + '\n')
    fd.close()
    val_acc.append(top1.ave)

    return top1.ave


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""
    if not args.cos_lr:
        if epoch in training_configurations[args.model]['changing_lr']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= training_configurations[args.model]['lr_decay_rate']
        print('lr:')
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

    else:
        for param_group in optimizer.param_groups:
            if epoch <= 10:
                param_group['lr'] = 0.5 * training_configurations[args.model]['initial_learning_rate']\
                                * (1 + math.cos(math.pi * epoch / training_configurations[args.model]['epochs'])) * (epoch - 1) / 10 + 0.01 * (11 - epoch) / 10
            else:
                param_group['lr'] = 0.5 * training_configurations[args.model]['initial_learning_rate']\
                                    * (1 + math.cos(math.pi * epoch / training_configurations[args.model]['epochs']))
        print('lr:')
        for param_group in optimizer.param_groups:
            print(param_group['lr'])


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
