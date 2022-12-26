import torch
import torch.nn as nn
from MIA.model import WhiteBoxAttackModel
from sparselearning.utils import  get_cifar100_dataloaders
from sparselearning.models import VGG16
from MIA.white_train import mia_train
import torch.backends.cudnn as cudnn
import torch.optim as optim
from MIA.white_eval import mia_evaluate,evaluate
import datetime
import os
from utils import  get_gradient_size
if not os.path.exists('./logs'): os.mkdir('./logs')

def load_model(args,file_path,model):
    param = torch.load(file_path,map_location='cpu')
    model_param = param['net']
    if args.manner == 'dp_sgd':
        m_param = {}
        for k, v in model_param.items():
            m_param[".".join(k.split('.')[1:])] = v
        model_param = m_param
    # model_param = param['model']['state_dict']
    model.load_state_dict(model_param)
    return model

def save_checkpoints(adversary,epoch,optimizer,file_name):
    obj = { 'epoch': epoch,
            'net': adversary.state_dict(),
            'optimizer': optimizer.state_dict(),
    }
    torch.save(obj,file_name)



from log import  print_and_log, setup_logger

models = {}

models['vgg-c'] = (VGG16, ['C', 100])

logger = None
cudnn.benchmark = True
cudnn.deterministic = True
import  argparse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    
    parser.add_argument('--manner', type=str, default="minmax", 
                        help='the defense manner')
    parser.add_argument('--mode', type=str, default="prune", 
                        help='the compress mode')
    
    randomhash = "-".join("-".join(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split(" ")).split(":"))
    parser.add_argument('--save', type=str, default=randomhash + '.pth',
                        help='path to save the final model')

    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--student', type=str, default='')

    parser.add_argument("--n_class", type=int, default=100, help="number of class in dataset using now")
    parser.add_argument('--density', type=float, default=0.05, help='The density of the overall sparse network.')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--datadir', type=str, default='/data/buffcal/tinyimagenet/tiny-imagenet-200/')
    parser.add_argument('--save-features', action='store_true', help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument("--minmax", action='store_true', help='If conbining with minmax strategy')
    parser.add_argument('--fp16', action='store_true', help='Run in fp16 mode.')

    args = parser.parse_args()
    device = 'cpu'
    setup_logger(args)
    print_and_log(args)

    if args.model not in models:
        print('You need to select an existing model via the --model argument. Available models include: ')
        for key in models:
            print('\t{0}'.format(key))
        raise Exception('You need to select a model')
    else:
        cls, cls_args = models[args.model]
        model = cls(*(cls_args + [args.save_features, args.bench])).to(device)

    if args.data == 'cifar100':
        train_loader, valid_loader, test_loader, known_loader, infset_loader, refer_loader, test_infset_loader = get_cifar100_dataloaders(
            args, args.valid_split, max_threads=args.max_threads)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ## initilize whitebox attackers
    gradient_size = get_gradient_size(model)
    total = gradient_size[0][0]//2 * gradient_size[0][1]//2
    adversary = WhiteBoxAttackModel(args.n_class,total).to(device)

    print_and_log(adversary)
    optimizer_mia = optim.Adam(adversary.parameters(), lr=0.001)

    
    file_path = "./checkpoints/vgg-c_0.1_cifar100_2022-11-28-17-45-56.pth"
    
    model = load_model(args,file_path,model)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5.0e-4, nesterov=True)
    evaluate(args, model, device, test_loader, is_test_set=False)
    size = len(refer_loader)

    best_mia_acc = 0
    for epoch in range(args.epochs):
        print_and_log("*" * 40)
        print_and_log("current epoch is {}/{}".format(epoch,args.epochs))
        print_and_log("*" * 40)

        train_private_enum = enumerate(zip(known_loader, refer_loader))
        privacy_loss, privacy_acc = mia_train(args, model, adversary, device, optimizer, \
                                                  train_private_enum, optimizer_mia,size=size, minmax=args.minmax)

        test_private_enum = enumerate(zip(infset_loader,test_infset_loader))
        inf_acc = mia_evaluate(args, model, adversary, optimizer, device, test_private_enum) 
        if inf_acc > best_mia_acc:
            best_mia_acc = inf_acc
            if args.density!=1 and args.manner!="mia_safecompress":
                save_checkpoints(adversary,epoch,optimizer_mia,'./whitemia_checkpoints/'+ \
                                 "_".join([args.manner,str(args.density),args.model,args.mode,args.save]))
            else:
                save_checkpoints(adversary,epoch,optimizer_mia,'./whitemia_checkpoints/'+ \
                                 "_".join([args.manner,str(args.density),args.model,args.save]))
                
    print_and_log("=" * 40)
    if args.density!=1 and args.manner!="mia_safecompress":
        param = torch.load('./whitemia_checkpoints/'+ "_".join([args.manner,str(args.density),args.model,args.mode,args.save]))
    else:
        param = torch.load('./whitemia_checkpoints/'+ "_".join([args.manner,str(args.density),args.model,args.save]))
       
    adversary.load_state_dict(param['net'])
    print_and_log("the best mia acc epoch:{}".format(param['epoch']))
    test_private_enum = enumerate(zip(infset_loader, test_infset_loader))
    inf_acc = mia_evaluate(args, model, adversary, optimizer, device, test_private_enum,is_test_set=True)
    

            
            

            
            

                
