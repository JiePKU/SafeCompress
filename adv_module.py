import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from itertools import cycle
from log import  print_and_log
import time
from MIA.entropy_regularization import EntropyLoss,ThresholdEntropyLoss,KLEntropyLoss,AguEntropyLoss
from utils import get_attack_input_data_for_whitebox

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self,pred,label):
        loss = self.ce(pred,label)
        return loss,torch.Tensor([0])

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, model, device, train_enum, optimizer, size, mask=None, num_batches=10000):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    entroys = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    first_id = -1

    if args.regularization != None:
        if args.regularization == "EntropyLoss":
            criterion = EntropyLoss()
        elif args.regularization == "ThresholdEntropyLoss":
            criterion = ThresholdEntropyLoss()
        elif args.regularization == "KLEntropyLoss":
            criterion = KLEntropyLoss(n_class=args.n_class)
        else:
            criterion = AguEntropyLoss(beta=0.08)
    else:
        criterion = CELoss()

    for batch_idx, (data, target) in train_enum:

        if isinstance(target, list):
            target = target[0]

        if first_id == -1:
            first_id = batch_idx

        data, target = data.to(device), target.to(device)


        data_time.update(time.time() - end)

        if args.fp16: data = data.half()
        optimizer.zero_grad()
        output = model(data)
        loss, entroy = criterion(output, target)
        # entroy = L2_Re(model,1e-4)
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 3))
        entroys.update(entroy.item(), data.size(0))
        losses.update(loss.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))
        top5.update(prec5.item(), data.size(0))

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if mask is not None:
            mask.step()
        else:
            optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if batch_idx % 100 == 0:
            print_and_log(
                '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | Entroy: {entroy:.4f} |top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=size,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    entroy=entroys.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if batch_idx - first_id >= num_batches:
            break
    return (losses.avg, top1.avg)


def evaluate(args, model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0

    with torch.no_grad():
        for data, target in test_loader:
            if isinstance(target,list):
                target = target[0]

            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()

            model.t = target
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print_and_log('\n{}: Classification average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Model Test evaluation' if is_test_set else 'Model Evaluation',
        test_loss, correct, n, 100. * correct / float(n)))

    return correct / float(n)


def adv_train(args, model, train_private_enum, \
                device, optimizer_mia, size, num_batchs, \
                    black_adversary, white_adversary,  optimizer):

    batch_time = AverageMeter()
    data_time = AverageMeter()

    loss_white = AverageMeter()
    loss_black = AverageMeter()

    top1_white = AverageMeter()
    top1_black = AverageMeter()


    black_adversary.train()
    white_adversary.train()


    model.eval()

    end = time.time()
    first_id = -1

    # when short dataloader is over, thus end
    for batch_idx, data in train_private_enum:

        ((tr_input, tr_target), (te_input, te_target)) = data

        if first_id == -1:
            first_id = batch_idx
        # measure data loading time


        ## for black-box and white-box mia

        data_time.update(time.time() - end)
        tr_input = tr_input.to(device)
        te_input = te_input.to(device)
        tr_target = tr_target.to(device)
        te_target = te_target.to(device)
        # compute output
        model_input = torch.cat((tr_input, te_input))
        if args.fp16: model_input = model_input.half()
        infer_input = torch.cat((tr_target, te_target))

        ## for white adversarial or both
        if args.adv_mode == 2 or args.adv_mode == 1:
            pred_outputs, gradients, losses_ = get_attack_input_data_for_whitebox(model, model_input, infer_input, \
                                                                                nn.CrossEntropyLoss(reduction="none"), optimizer)
        elif  args.adv_mode == 0:
            with torch.no_grad():
                  pred_outputs = model(model_input)
        
        one_hot_tr = torch.from_numpy((np.zeros((infer_input.size(0), args.n_class)) - 1)).cuda().type(torch.cuda.FloatTensor)
        infer_input_one_hot = one_hot_tr.scatter_(1, infer_input.type(torch.cuda.LongTensor).view([-1, 1]).data, 1)

        attack_model_input = pred_outputs  # torch.cat((pred_outputs,infer_input_one_hot),1)
        v_is_member_labels = torch.from_numpy(
            np.reshape(np.concatenate((np.ones(tr_input.size(0)), np.zeros(te_input.size(0)))), [-1, 1])).cuda().type(torch.cuda.FloatTensor)

        r = np.arange(v_is_member_labels.size()[0]).tolist()
        random.shuffle(r)
        attack_model_input = attack_model_input[r]
        v_is_member_labels = v_is_member_labels[r]
        infer_input_one_hot = infer_input_one_hot[r]

        ## for white adversarial or both
        if args.adv_mode == 1:
            gradients = gradients[r]
            losses_ = losses_[r]
            white_member_output = white_adversary(attack_model_input, losses_, gradients, infer_input_one_hot)
            white_loss = F.binary_cross_entropy(white_member_output, v_is_member_labels)
            loss = white_loss

        if args.adv_mode == 2 :
            gradients = gradients[r]
            losses_ = losses_[r]

            black_member_output = black_adversary(attack_model_input, infer_input_one_hot)
            white_member_output = white_adversary(attack_model_input, losses_, gradients, infer_input_one_hot)

            black_loss = F.binary_cross_entropy(black_member_output, v_is_member_labels)
            white_loss = F.binary_cross_entropy(white_member_output, v_is_member_labels)

            loss = black_loss + white_loss

        elif args.adv_mode == 0:
            black_member_output = black_adversary(attack_model_input, infer_input_one_hot)
            black_loss = F.binary_cross_entropy(black_member_output, v_is_member_labels)
            loss = black_loss

        # measure accuracy and record loss
        if args.adv_mode == 1:
            white_prec1 = np.mean((white_member_output.data.cpu().numpy() > 0.5) == v_is_member_labels.data.cpu().numpy())
            loss_white.update(white_loss.item(), model_input.size(0))
            top1_white.update(white_prec1, model_input.size(0))

        elif args.adv_mode == 0:
            black_prec1 = np.mean((black_member_output.data.cpu().numpy() > 0.5) == v_is_member_labels.data.cpu().numpy())
            loss_black.update(black_loss.item(), model_input.size(0))
            top1_black.update(black_prec1, model_input.size(0))

        elif args.adv_mode == 2:
            black_prec1 = np.mean((black_member_output.data.cpu().numpy() > 0.5) == v_is_member_labels.data.cpu().numpy())
            loss_black.update(black_loss.item(), model_input.size(0))
            top1_black.update(black_prec1, model_input.size(0))

            white_prec1 = np.mean((white_member_output.data.cpu().numpy() > 0.5) == v_is_member_labels.data.cpu().numpy())
            loss_white.update(white_loss.item(), model_input.size(0))
            top1_white.update(white_prec1, model_input.size(0))

        # compute gradient and do SGD step

        optimizer_mia.zero_grad()
        if args.fp16:
            optimizer_mia.backward(loss)
        else:
            loss.backward()

        optimizer_mia.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if batch_idx % 10 == 0:
            print_and_log(
                '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | black Loss: {black_loss:.4f} | white Loss: {white_loss:.4f} |  top1_black: {top1_black: .4f} | top1_white: {top1_white: .4f} |'.format(
                    batch=batch_idx,
                    size=size,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    black_loss = loss_black.avg,
                    white_loss = loss_white.avg,
                    top1_black = top1_black.avg,
                    top1_white = top1_white.avg,
                ))
        
        if batch_idx - first_id >= num_batchs:
            break
        
    return top1_black.avg, top1_white.avg


def train_privately(args, model, device, train_enum,
                    optimizer, size, black_adversary,
                    white_adversary,
                    mask=None, num_batches=10000):

    model.train()
    white_adversary.eval()
    black_adversary.eval()


    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    adv_losses = AverageMeter()
    entroys = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    first_id = -1

    if args.regularization != None:
        if args.regularization == "EntropyLoss":
            criterion = EntropyLoss()
        elif args.regularization == "ThresholdEntropyLoss":
            criterion = ThresholdEntropyLoss()
        elif args.regularization == "KLEntropyLoss":
            criterion = KLEntropyLoss(n_class=args.n_class)
        else:
            criterion = AguEntropyLoss(beta=0.08)
    else:
        criterion = CELoss()

    for batch_idx, (data, target) in train_enum:

        if isinstance(target, list):
            target, attri_target = target[0],target[1]

        if first_id == -1:
            first_id = batch_idx

        data, target = data.to(device), target.to(device)
        if args.adv_mode == 2 or args.adv_mode==1:
            pred_outputs, gradients, losses_ = get_attack_input_data_for_whitebox(model, data, target,
                                                                         nn.CrossEntropyLoss(reduction="none"), optimizer)

        model.train()
        data_time.update(time.time() - end)

        if args.fp16: data = data.half()
        optimizer.zero_grad()
        output = model(data)


        one_hot_label = torch.from_numpy(np.zeros((data.size()[0],args.n_class))-1).to(device).float()
        one_hot_label = one_hot_label.scatter_(1,target.type(torch.cuda.LongTensor).view([-1, 1]).data,1)

        mia_label = torch.from_numpy(np.ones(data.size()[0])).to(device).float().reshape(-1,1)

        if args.adv_mode == 2:
            white_mia_out = white_adversary(output, losses_, gradients, one_hot_label)
            black_mia_out = black_adversary(output,one_hot_label)
        if args.adv_mode==1:
            white_mia_out = white_adversary(output, losses_, gradients, one_hot_label)
        elif args.adv_mode==0:
            black_mia_out = black_adversary(output,one_hot_label)


        loss1, entroy = criterion(output, target)
        
        if args.adv_mode == 2:
            white_mia_loss = F.binary_cross_entropy(white_mia_out, mia_label)
            black_mia_loss = F.binary_cross_entropy(black_mia_out, mia_label)
            loss = loss1 - 0.1 * (black_mia_loss + white_mia_loss)
            adv_loss = black_mia_loss + white_mia_loss
        if args.adv_mode==1:
            white_mia_loss = F.binary_cross_entropy(white_mia_out, mia_label)
            loss = loss1 - 0.1 * (white_mia_loss)
            adv_loss = white_mia_loss
        elif args.adv_mode==0:
            black_mia_loss = F.binary_cross_entropy(black_mia_out, mia_label)
            loss = loss1 - 0.1 * (black_mia_loss)
            adv_loss = black_mia_loss   


        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 3))
        entroys.update(entroy.item(), data.size(0))
        losses.update(loss1.item(), data.size(0))
        adv_losses.update(adv_loss.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))
        top5.update(prec5.item(), data.size(0))

        if args.fp16: optimizer.backward(loss)
        else:
            loss.backward()

        if mask is not None:
            mask.step()
        else:
            optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if batch_idx % 100 == 0:
            print_and_log(
                '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | | adv Loss: {adv_loss:.4f}  | Entroy: {entroy:.4f} |top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=size,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    adv_loss = adv_losses.avg,
                    entroy=entroys.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))

        if batch_idx - first_id >= num_batches:
            break

    return (losses.avg, top1.avg)