import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from itertools import cycle
from log import  print_and_log
import time
from utils import  get_attack_input_data_for_whitebox

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

def mia_train(args, model, adversary, device, optimizer,\
                      train_private_enum, optimizer_mia, size ,minmax = False, num_batchs=1000):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    adversary.train()
    model.eval()
    # train inference model
    # from itertools import cycle
    # for batch_idx, (data, target) in enumerate(zip(known_loader, cycle(refer_loader))):
    end = time.time()
    first_id = -1
    # when short dataloader is over, thus end
    for batch_idx, ((tr_input, tr_target), (te_input, te_target)) in  train_private_enum:
        # measure data loading time

        if first_id == -1:
            first_id = batch_idx

        data_time.update(time.time() - end)
        tr_input = tr_input.to(device)
        te_input = te_input.to(device)
        tr_target = tr_target.to(device)
        te_target = te_target.to(device)
        # compute output
        model_input = torch.cat((tr_input, te_input))

        if args.fp16: model_input = model_input.half()

        infer_input = torch.cat((tr_target, te_target))

        ## we get extra infor from model in white-box setting
        pred_outputs, gradients, losses_ = get_attack_input_data_for_whitebox(model,model_input,infer_input,nn.CrossEntropyLoss(reduction="none"), optimizer)

        ## to one-hot form
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
        gradients = gradients[r]
        losses_ = losses_[r]

        member_output = adversary(attack_model_input,losses_,gradients,infer_input_one_hot)

        loss = F.binary_cross_entropy(member_output, v_is_member_labels)

        # measure accuracy and record loss
        prec1 = np.mean((member_output.data.cpu().numpy() > 0.5) == v_is_member_labels.data.cpu().numpy())
        losses.update(loss.item(), model_input.size(0))
        top1.update(prec1, model_input.size(0))

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

        if batch_idx - first_id >= num_batchs:
            break

        # plot progress
        if batch_idx % 10 == 0:
            print_and_log(
                '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
                    batch=batch_idx,
                    size=size,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                ))

    return (losses.avg, top1.avg)