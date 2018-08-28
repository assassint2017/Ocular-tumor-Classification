"""

训练脚本
"""

import os
from time import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from net.DenseNet import DenseNet
from dataset.dataset import Dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Hyper parameters
epoch_num = 601
batch_size = 32
lr = 1e-4  # learning rate
weight_decay = 1e-5
cv_data_dir = './cv_data/'

lbl_list = []
yz_list = []
for fold in os.listdir(cv_data_dir):
    lbl_list.append(os.path.join(cv_data_dir, fold, 'lbl'))
    yz_list.append(os.path.join(cv_data_dir, fold, 'yz'))

# loss function
loss_func = nn.CrossEntropyLoss()

# train the network
start = time()

precision_list = []
accuracy_list = []
recall_list = []
F1_score_list = []

for fold_index in range(5):

    # define network
    # net = VGG().cuda()
    net = DenseNet([3, 3, 4, 4], 16).cuda()
    net = nn.DataParallel(net)

    # define optimizer
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    # opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)

    # define learning rate decay
    lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [150])

    # define dataset size
    test_ds_size = 26 if fold_index is not 4 else 28
    train_ds_size = 132 - test_ds_size

    # define dataset
    val_ds = Dataset(lbl_list[fold_index], yz_list[fold_index], istraining=False)
    val_dl = DataLoader(val_ds, batch_size)

    temp_lbl_list = lbl_list.copy()
    temp_yz_list = yz_list.copy()

    del temp_lbl_list[fold_index]
    del temp_yz_list[fold_index]

    train_ds = Dataset(temp_lbl_list, temp_yz_list, istraining=True)
    train_dl = DataLoader(train_ds, batch_size, True)

    best_val_acc = 0

    for epoch in range(epoch_num):

        lr_decay.step()

        for step, (data, target) in enumerate(train_dl, 1):

            data, target = data.cuda(), target.cuda()
            target = target.squeeze()

            outputs = net(data)

            loss = loss_func(outputs, target)

            opt.zero_grad()

            loss.backward()

            opt.step()

            if step % 3 is 0:

                print('fold:{} epoch:{}, step:{}, loss:{:.3f}, time:{:.1f} min'
                      .format(fold_index, epoch, step, loss.item(), (time() - start) / 60))

        # 每训练一定epoch测试一下精度并保存模型参数
        if epoch % 10 is 0 and epoch is not 0:
            train_acc, val_acc = 0, 0

            net.eval()
            with torch.no_grad():

                # 测试训练集上的精度
                for data, target in train_dl:
                    data, target = data.cuda(), target.cuda()
                    target = target.squeeze()

                    outputs = net(data)

                    train_acc += sum(torch.max(outputs, 1)[1].detach().cpu().numpy() == target.detach().cpu().numpy())

                # 测试验证训练集上的精度
                for data, target in val_dl:
                    data, target = data.cuda(), target.cuda()
                    target = target.squeeze()

                    outputs = net(data)

                    val_acc += sum(torch.max(outputs, 1)[1].detach().cpu().numpy() == target.detach().cpu().numpy())

            train_acc /= train_ds_size
            val_acc /= test_ds_size

            if val_acc > best_val_acc:

                best_val_acc = val_acc

                # 保存在每一折中验证集acc最高的模型
                torch.save(net.state_dict(), './module/net-fold{}.pth'.format(fold_index))

            print('-------------------------------')
            print('val_acc:{:.3f}%, train_acc:{:.3f}%'.format(val_acc * 100, train_acc * 100))
            print('-------------------------------')

            net.train()

    # 在每一折训练结束后计算评价指标,这里约定把lbl称为正样本
    net = torch.nn.DataParallel(DenseNet([3, 3, 4, 4], 16)).cuda()
    net.load_state_dict(torch.load('./module/net-fold' + str(fold_index) + '.pth'))
    net.eval()

    num_lbl = 14
    num_yz = 12 if fold_index is not 4 else 14

    TPTN, TPFP = 0, 0

    with torch.no_grad():
        for data, target in val_dl:
            data, target = data.cuda(), target.cuda()
            target = target.squeeze()
            outputs = net(data)
            TPTN += sum(torch.max(outputs, 1)[1].detach().cpu().numpy() == target.detach().cpu().numpy())
            TPFP += sum(torch.max(outputs, 1)[1].detach().cpu().numpy() == 1)

    TP = ((TPTN + TPFP) - num_yz) // 2
    TN = TPTN - TP
    FP = num_yz - TN
    FN = num_lbl - TP

    accuracy = (TP + TN) / (TP + TN + FN + FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = 2 * (precision * recall) / (precision + recall)

    print('++++++++++++++++++++++++')
    print('accurary:{:.3f}, precision:{:.3f}, recall:{:.3f}, F1:{:.3f}'.format(accuracy, precision, recall, F1_score))
    print('++++++++++++++++++++++++')

    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    F1_score_list.append(F1_score)

# 训练结束后打印最终的评价指标
print('end of training')

print('accuracy', accuracy_list)
print('precision', precision_list)
print('recall', recall_list)
print('F1_score', F1_score_list)

print('---------------------------')

print('mean accuracy:{:.3f}'.format(sum(accuracy_list) / len(accuracy_list)))
print('mean precision:{:.3f}'.format(sum(precision_list) / len(precision_list)))
print('mean recall:{:.3f}'.format(sum(recall_list) / len(recall_list)))
print('mean F1_score:{:.3f}'.format(sum(F1_score_list) / len(F1_score_list)))
