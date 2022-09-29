from one_stage_model import *
from two_stage_model import *
from data_transform import *
from utils import train_nolabel
import math

base_lr = 0.001
epoches = 501
# lr_step = 70
outputdim = 60
dim = 1

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        ret = output1 * output2
        ret = torch.sum(ret, dim=1).view(-1, 1)
        ret = 1 - ret
        ret = ret.pow(2)
        dc = ret + 0.25 * (torch.abs(output1-output2)/math.sqrt(2))
        loss_contrastive = torch.mean((1-label) * dc + (label) * torch.pow(torch.clamp(self.margin - dc, min=0.0), 2))
        return loss_contrastive

net1 = encoder_plus_add(dim,outputdim)
net2 = decoder_plus_add(outputdim)

optimizer1 = torch.optim.Adam(net1.parameters(), lr=base_lr)


criterion = ContrastiveLoss()
# def adjust_lr(optimizer, epoch):
#     lr = base_lr*(0.1**(epoch//lr_step))
#     for parameter in optimizer.param_groups:
#         parameter['lr'] = lr

print(" ####Start training  ####")

for epoch in range(epoches):
    train_nolabel(net1,net2,L_train_data_1,L_test_data_1,epoch,optimizer1,criterion)

print("Done!")
