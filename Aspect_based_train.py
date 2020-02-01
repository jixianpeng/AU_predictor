# %%
from data_aspect import *
from torch import optim, nn
import torch
import itertools
# %%
import importlib
import Aspect_Based_Fusion

importlib.reload(Aspect_Based_Fusion)
from Aspect_Based_Fusion import *


# 又要reload又要from load的实现方式，jupyter不那么智能
def re_target(x):
    if x == -1:
        return 0
    else:
        return x
def hard_sample(loss, number):
    sort_index = torch.argsort(loss)
    temp_loss = loss[sort_index]
    # part_loss = temp_loss[-number*100::]
    part_loss = temp_loss[-number*2::]
    part_mean_loss = torch.mean(part_loss)
    return part_mean_loss
cc = ResNet().cuda()
ss = Landmark_diff_module().cuda()
aq = Aspect_Query().cuda()
re_training = True
if re_training == False:
    old_static_state = torch.load('./cc')
    cc.load_state_dict(old_static_state)
    old_static_state = torch.load('./ss')
    ss.load_state_dict(old_static_state)
    old_static_state = torch.load('./aq')
    aq.load_state_dict(old_static_state)
crition=nn.CrossEntropyLoss()
# crition = nn.CrossEntropyLoss(reduce=False)
# crition=nn.MultiLabelSoftMarginLoss()
# crition=nn.MultiLabelMarginLoss()
op = optim.Adam(itertools.chain(cc.parameters(), ss.parameters(), aq.parameters()), )
print('load model')

total_P = np.zeros((8))  # 总共预测的P
TP = np.zeros((8))  # 预测的P中正确的个数
one_TP = np.zeros((8))
total_D = np.zeros((8))  # 数据中总共有的P
T = 0
acc = 0

for i in range(1000000):
    temp_data, temp_label, temp_landmark = get_batch(bacth_size)
    # the reason for not using Filp is that it is easy for img but not for landmark
    temp_label_ = torch.LongTensor([re_target(l) for l in temp_label]).cuda()
    temp_data = torch.FloatTensor(temp_data).cuda()
    temp_landmark = torch.FloatTensor(temp_landmark).cuda()

    cnn_feature = cc(temp_data)
    landmark_feature = ss(temp_landmark)
    out = aq(cnn_feature, landmark_feature)
    error_number = bacth_size * 8 - torch.sum(torch.argmax(out, dim=1) == temp_label_).detach().cpu().numpy()
    loss1 = crition(out, temp_label_)
    # loss1=hard_sample(loss1,error_number)
    loss = loss1
    print(i,error_number)

    op.zero_grad()
    loss.backward()
    op.step()

    TP += np.sum(np.reshape((torch.argmax(out, dim=1) * temp_label_).detach().cpu().numpy(), (bacth_size, 8)), axis=0)
    total_P += np.sum(np.reshape((torch.argmax(out, dim=1)).detach().cpu().numpy() + 0.0001, (bacth_size, 8)), axis=0)
    total_D += np.sum(np.reshape(temp_label, (bacth_size, 8)) + 0.0001, axis=0)
    acc += torch.sum(torch.argmax(out, dim=1) == temp_label_)
    if i % 30 == 0 and i != 0:
        print(i)
        Precise = TP / total_P
        Recall = TP / total_D
        total_D = total_D.astype(np.int)
        Precise = Precise[np.where(total_D != 0)]
        Recall = Recall[np.where(total_D != 0)]
        F1 = np.mean(2 * (Precise * Recall) / (Precise + Recall + 0.001))

        acc = acc.detach().cpu().numpy() / (bacth_size * 30 * 8 * 1.0)

        print('Percormance:', 0.5 * acc + 0.5 * F1, 'ACC:', acc, 'F1:', F1)
        print('TP:', TP)
        print('TD:', total_D.astype(np.int))
        print('Precise:', list(Precise))
        print('Recall：', list(Recall))
        print('loss:', loss)

        total_P = np.zeros((8))  # 总共预测的P
        TP = np.zeros((8))  # 预测的P中正确的个数
        one_TP = np.zeros((8))
        total_D = np.zeros((8))  # 数据中总共有的P
        T = 0
        acc = 0
    if i % 400 == 0 and i != 0:
        torch.save(cc.state_dict(), './cc')
        torch.save(ss.state_dict(), './ss')
        torch.save(aq.state_dict(), './aq')






