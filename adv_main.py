from __future__ import print_function

import os
import argparse
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import sparselearning
from sparselearning.core import Masking, CosineDecay, LinearDecay
from sparselearning.models import  VGG16
from sparselearning.utils import get_cifar100_dataloaders, prepare_dataset
import warnings
import datetime

from MIA import Adversary
from MIA.model import WhiteBoxAttackModel

from adv_module import train, adv_train, train_privately, evaluate
from utils import get_gradient_size

warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True

if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

models = {}
models['vgg-c'] = (VGG16, ['C', 100])
from log import print_and_log, setup_logger


def save_checkpoints(model,epoch,file_name):
    obj = { 'epoch':epoch,
            'net':model.state_dict(),
    }
    torch.save(obj,file_name)

def load_checkpoints(model,file_name):
    param = torch.load(file_name)
    model.load_state_dict(param['net'])
    print(param['epoch'])
    return model

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--multiplier', type=int, default=1, metavar='N',
                        help='extend training time by multiplier times')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--optimizer', type=str, default='sgd', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    randomhash = "-".join("-".join(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split(" ")).split(":"))
    parser.add_argument('--save', type=str, default=randomhash + '.pth',
                        help='path to save the final model')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--datadir', type=str, default='./data/tiny-imagenet-200/')
    parser.add_argument('--decay_frequency', type=int, default=25000)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--fp16', action='store_true', help='Run in fp16 mode.')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--l2', type=float, default=5.0e-4)
    parser.add_argument('--iters', type=int, default=1, help='How many times the model should be run after each other. Default=1')
    parser.add_argument('--save-features', action='store_true', help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')

    # training inference attack model strategy minmax adv
    parser.add_argument("--n_class",type=int, default=100,help="number of class in dataset using now")

    # pretrain classification model
    parser.add_argument("--pretrain_epoch", type=int, default=200, help="pretrain the model effeciently")


    parser.add_argument("--adv_mode", type=int, default=2, help="0 for black 1 for white and 2 for both")
    # regularization term
    parser.add_argument("--regularization", type=str, default=None, help="EntropyLoss,AguEntropyLoss,ThresholdEntropyLoss,KLEntropyLoss")

    parser.add_argument("--attri", action='store_true',
                        help="EntropyLoss,AguEntropyLoss,ThresholdEntropyLoss,KLEntropyLoss")

    # ITOP settings
    sparselearning.core.add_sparse_args(parser)

    args = parser.parse_args()
    setup_logger(args)
    print_and_log(args)

    if args.fp16:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print_and_log('\n\n')
    print_and_log('='*80)
    torch.manual_seed(args.seed)
    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i+1, args.iters))

        if args.data == 'cifar100':
            train_loader, valid_loader, test_loader, known_loader, infset_loader, refer_loader, test_infset_loader = get_cifar100_dataloaders(args, args.valid_split, max_threads=args.max_threads)

        if args.model not in models:
            print('You need to select an existing model via the --model argument. Available models include: ')
            for key in models:
                print('\t{0}'.format(key))
            raise Exception('You need to select a model')
        else:
            cls, cls_args = models[args.model]
            cls_args[1] = args.n_class
            model = cls(*(cls_args + [args.save_features, args.bench])).to(device)

        print_and_log(model)
        print_and_log('=' * 60)
        print_and_log(args.model)
        print_and_log('=' * 60)

        # MIA model
        ## initilize whitebox attackers
        black_adversary = Adversary(args.n_class).to(device)

        # print(black_adversary)

        gradient_size = get_gradient_size(model)
        total = gradient_size[0][0] // 2 * gradient_size[0][1] // 2
        white_adversary = WhiteBoxAttackModel(args.n_class, total).to(device)

        # print(white_adversary)

        print_and_log('=' * 60)
        print_and_log('Prune mode: {0}'.format(args.death))
        print_and_log('Growth mode: {0}'.format(args.growth))
        print_and_log('Redistribution mode: {0}'.format(args.redistribution))
        print_and_log('=' * 60)

        if args.adv_mode==0:
            optimizer_mia = optim.Adam([{'params':black_adversary.parameters(),
                                    'lr':0.001,
                                    }])
        elif args.adv_mode==1:
            optimizer_mia = optim.Adam([{'params':white_adversary.parameters(),
                                    'lr':0.001}])
        elif args.adv_mode==2:
            optimizer_mia = optim.Adam([{'params': black_adversary.parameters(),
                                        'lr': 0.001,
                                        },
                                    {'params': white_adversary.parameters(),
                                        'lr': 0.001}])

        optimizer = None
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2, nesterov=True)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        else:
            print('Unknown optimizer: {0}'.format(args.optimizer))
            raise Exception('Unknown optimizer.')

        milestone = [int(args.epochs / 2) * args.multiplier, int(args.epochs * 3 / 4) * args.multiplier]

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones= milestone, last_epoch=-1)

        if args.fp16:
            print('FP16')
            optimizer = FP16_Optimizer(optimizer,
                                       static_loss_scale = None,
                                       dynamic_loss_scale = True,
                                       dynamic_loss_args = {'init_scale': 2 ** 16})
            model = model.half()

        mask = None
        if args.sparse:
            decay = CosineDecay(args.death_rate, len(train_loader)*(args.epochs*args.multiplier))
            mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay, growth_mode=args.growth,
                           redistribution_mode=args.redistribution, args=args)
            mask.add_module(model, sparse_init=args.sparse_init, density=args.density)

        best_acc = 0.0
        size = len(train_loader)
        size_r = len(refer_loader)
        time = size_r // 2

        for epoch in range(args.start_epoch, args.epochs*args.multiplier + 1):
            print_and_log("*"*40)
            print_and_log("current epoch is {}".format(epoch))
            print_and_log("*" * 40)
            train_enum = enumerate(train_loader)
            train_private_enum = enumerate(zip(known_loader, refer_loader))

            if epoch < args.pretrain_epoch:
                _ , __ = train(args, model, device, train_enum , optimizer, size, mask)

            else:
                for i in range(size//2):
                    num_batches = 1
                    black_acc, white_acc = adv_train(args, model, train_private_enum, \
                                                          device,optimizer_mia, size, num_batches, \
                            black_adversary, white_adversary,  optimizer)

                    train_loss, train_acc = train_privately(args, model, device, train_enum, optimizer, size, black_adversary, white_adversary,  mask, num_batches)

                    if i % 10 == 0:
                        print_and_log(
                            'privacy black acc: {}'.format(black_acc) + ' privacy white acc: {}'.format(white_acc) + ' privacy attri acc: {}'.format(attri_acc)  + ' train acc: {}'.format(train_acc))

                    if (i+1)% time ==0:
                        train_private_enum = enumerate(zip(known_loader, refer_loader))

            lr_scheduler.step()
            if args.valid_split > 0.0:
                val_acc = evaluate(args, model, device, valid_loader)
                if val_acc > best_acc:
                    print('Saving model')
                    best_acc = val_acc
                    save_checkpoints(model,
                                     epoch,f"./checkpoints/{args.model}_{args.density}_{args.data}_{args.save}")


            print('Testing model and adversity')
            save_checkpoints(model, epoch, f"./checkpoints/_{args.density}_last.pth")
        
        model = load_checkpoints(model, f"./checkpoints/{args.model}_{args.density}_{args.data}_{args.save}")
        print_and_log("\nIteration end: {0}/{1}\n".format(i+1, args.iters))
        
        evaluate(args, model, device, test_loader, is_test_set=True)
        
        layer_fired_weights, total_fired_weights = mask.fired_masks_update()
        for name in layer_fired_weights:
            print_and_log('The final percentage of fired weights in the layer', name, 'is:', layer_fired_weights[name])
        print_and_log('The final percentage of the total fired weights is:', total_fired_weights)


if __name__ == '__main__':
   main()
