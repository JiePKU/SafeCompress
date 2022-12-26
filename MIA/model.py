import torch
import torch.nn as nn


"""
This model is referred to paper:

"Machine Learning with Membership Privacy using Adversarial Regularization"

More detail can be found in:
https://dl.acm.org/doi/abs/10.1145/3243734.3243855

this code is implemented in 2022.01.03 

version : v1

"""

class Adversary(nn.Module):   #  black-box setting
    def __init__(self,n_class=100):
        super(Adversary, self).__init__()
        self.n_class = n_class

        # for prediction
        self.pred_fc = nn.Sequential(nn.Linear(self.n_class,1024),
                                     nn.ReLU(),
                                     nn.Linear(1024,512),
                                     nn.ReLU(),
                                     nn.Linear(512,64),
                                     nn.ReLU())
        # for label
        self.label_fc = nn.Sequential(nn.Linear(self.n_class,512),
                                      nn.ReLU(),
                                      nn.Linear(512,64),
                                      nn.ReLU())

        # fuse layer
        self.class_layer = nn.Sequential(nn.Linear(128,256),
                                        nn.ReLU(),
                                        nn.Linear(256,64),
                                        nn.ReLU(),
                                        nn.Linear(64,1))

        # init weight
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                nn.init.normal_(self.state_dict()[key], std=0.01)

            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def forward(self,x,y):
        # x should be softmax output

        x1 = self.pred_fc(x) # B C
        x2 = self.label_fc(y)
        x12 = torch.cat([x1,x2],dim=1)
        # x12 = self.bn(x12)
        out = self.class_layer(x12)
        out = torch.sigmoid(out)
        return out

    def init_weights(self,m):
        if isinstance(m,nn.Linear):
            m.weight.data.normal_(0,0.01)
            if m.bias.data is not None:
                m.bias.data.fill_(0)


# class WhiteBoxAttackModel(nn.Module):
#     def __init__(self, class_num, total):
#         super(WhiteBoxAttackModel, self).__init__()

#         self.Output_Component = nn.Sequential(
#             nn.Dropout(p=0.2),
#             nn.Linear(class_num, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#         )

#         self.Loss_Component = nn.Sequential(
#             nn.Dropout(p=0.2),
#             nn.Linear(1, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#         )

#         self.Gradient_Component = nn.Sequential(
#             nn.Dropout(p=0.2),
#             nn.Conv2d(1, 1, kernel_size=5, padding=2),
#             nn.BatchNorm2d(1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Flatten(),
#             nn.Dropout(p=0.2),
#             nn.Linear(total, 256),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#         )

#         self.Label_Component = nn.Sequential(
#             nn.Dropout(p=0.2),
#             nn.Linear(class_num, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#         )

#         self.Encoder_Component = nn.Sequential(
#             nn.Dropout(p=0.2),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#         )
#         # init weight
#         self.apply(self.weights_init)

#     @staticmethod
#     def weights_init(m):
#         if isinstance(m, nn.Conv2d):
#             nn.init.normal_(m.weight.data)
#             m.bias.data.fill_(0)
#         elif isinstance(m, nn.Linear):
#             nn.init.xavier_normal_(m.weight)
#             nn.init.constant_(m.bias, 0)

#     def forward(self, output, loss, gradient, label):
#         Output_Component_result = self.Output_Component(output)
#         Loss_Component_result = self.Loss_Component(loss)
#         Gradient_Component_result = self.Gradient_Component(gradient)
#         Label_Component_result = self.Label_Component(label)

#         final_inputs = torch.cat(
#             (Output_Component_result, Loss_Component_result, Gradient_Component_result, Label_Component_result), 1)
#         final_result = self.Encoder_Component(final_inputs)

#         final_result = torch.sigmoid(final_result)

#         return final_result



class WhiteBoxAttackModel(nn.Module):
    def __init__(self, class_num, total):
        super(WhiteBoxAttackModel, self).__init__()

        self.Output_Component = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(class_num, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.Loss_Component = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.Gradient_Component = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv2d(1, 1, kernel_size=5, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(total, 256),
            nn.ReLU(),
            #nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.Label_Component = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(class_num, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.Encoder_Component = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # init weight
        self.apply(self.weights_init)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data)
            m.bias.data.fill_(0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, output, loss, gradient, label):
        Output_Component_result = self.Output_Component(output)
        Loss_Component_result = self.Loss_Component(loss)
        Gradient_Component_result = self.Gradient_Component(gradient)
        Label_Component_result = self.Label_Component(label)

        final_inputs = torch.cat(
            (Output_Component_result, Loss_Component_result, Gradient_Component_result, Label_Component_result), 1)
        final_result = self.Encoder_Component(final_inputs)

        final_result = torch.sigmoid(final_result)

        return final_result

# if __name__=="__main__":
#     ad = Adversary().train()
#     x = torch.randn([256,100])
#     index = torch.randint(0,100,[256]).unsqueeze(dim=1)
#     y = torch.zeros([256,100]).scatter_(value=1,index=index,dim=1)
#     out = ad(x,y)
#     print(out)
