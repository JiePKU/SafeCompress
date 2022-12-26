
## this script is created for attack test before the model update
import torch
import  torch.nn as nn


def get_gradient_size(model):
    gradient_size = []
    gradients = reversed(list(model.named_parameters()))
    for name, param in gradients:
        if "weight" in name:
            gradient_size.append(param.shape)
            break

    del gradients
    # here just one tuple
    return gradient_size


def get_attack_input_data_for_whitebox(model, input, target, criteria,optimizer):
    model.eval()
    output = model(input)  # B class_number
    losses = criteria(output,target)

    gradients = []
    for loss in losses:
        loss.backward(retain_graph=True)

        gradient_list = reversed(list(model.named_parameters()))

        for name, param in gradient_list:
            if "weight" in name:
                gradient = param.grad.clone()
                gradient = gradient.unsqueeze_(0)
                gradients.append(gradient.unsqueeze_(0))
                break
        ## in pytorch the gradient is summed 
        optimizer.zero_grad()
        
        del gradient_list

    ## we do not contain the target output
    gradients = torch.cat(gradients,dim=0) # B 1 H W
    losses = losses.unsqueeze(dim=1).detach()
    return output,gradients,losses


