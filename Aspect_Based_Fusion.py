import torch
import torch.nn as nn
import torch.nn.functional as F
bacth_size=1
cnn_kernal_size=96
landmark_init=73*2
landmark_final=cnn_kernal_size
aspect_number=64
aspect_dim=96
# from data_handler_boder import *
# temp_data,temp_label,temp_landmark=get_batch(20)


# from torchvision import models
# resnet=models.resnet18(pretrained=True)



#刚开始的时候尝试了resnet，失去了局部特性之后，对遮挡不鲁棒，且由于全脸建模，姿态多，致使数据稀疏，效果不好

# class ResidualBlock(nn.Module):
#     def __init__(self, inchannel, outchannel, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.left = nn.Sequential(
#             nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(outchannel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(outchannel)
#         )
#         self.shortcut = nn.Sequential()
#         if stride != 1 or inchannel != outchannel:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(outchannel)
#             )
#     def forward(self, x):
#         out = self.left(x)
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
# class ResNet(nn.Module):
#     def __init__(self, ResidualBlock=ResidualBlock, num_classes=10):
#         super(ResNet, self).__init__()
#         self.inchannel = 2
#         self.layer1 = self.make_layer(ResidualBlock, 16,  2, stride=2)
#         self.layer2 = self.make_layer(ResidualBlock, 32,  2, stride=2)
#         self.layer3 = self.make_layer(ResidualBlock, 64, 2, stride=2)
#         self.layer4 = self.make_layer(ResidualBlock, 128, 1, stride=2)
#         self.layer5 = self.make_layer(ResidualBlock, 64, 1, stride=2)
#         self.to_memory = nn.Sequential(nn.Linear(64, aspect_dim),nn.Tanh())
#     def make_layer(self, block, channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]，单个block段内都是一样的stride。
#         layers = []
#         for stride in strides:
#             layers.append(block(self.inchannel, channels, stride))
#             self.inchannel = channels
#         #收集参数的作用，就是将剩下的东西做成列表，Sequential的输入能够按照列表输入，因为平时的用法本来就是放到其收集参数当中去了
#         return nn.Sequential(*layers)
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.layer5(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(1,out.size(0),-1)
#         static_texture_memory = self.to_memory(out)
#         return static_texture_memory

#实验结果显示，局部模型比全局cnn好，对au15，au20能预测出
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        self.basic_layer=nn.Sequential(nn.Conv2d(in_channels=2,out_channels=16,kernel_size=3,stride=1),
                                       # nn.BatchNorm2d(16),
                                       nn.ELU(),
                                       nn.MaxPool2d(kernel_size=2, stride=2),

                                       nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1),
                                       nn.ELU(),
                                       nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
                                       nn.BatchNorm2d(64),
                                       nn.ELU(),
                                       nn.MaxPool2d(kernel_size=2, stride=2),

                                       # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
                                       nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
                                       nn.ELU(),
                                       nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1),
                                       nn.BatchNorm2d(128),
                                       nn.Sigmoid(),
                                       nn.MaxPool2d(kernel_size=2, stride=2),

                                       nn.Conv2d(in_channels=128, out_channels=cnn_kernal_size, kernel_size=1, stride=1),
                                       nn.Sigmoid(),
                                       nn.Conv2d(in_channels=cnn_kernal_size, out_channels=cnn_kernal_size, kernel_size=1, stride=1),
                                       nn.BatchNorm2d(cnn_kernal_size),
                                       nn.Sigmoid(),
                                       )
        self.cnn_serial=nn.GRU(cnn_kernal_size,aspect_dim,batch_first=True)
    def forward(self, input):
        pic=input
        pic=self.basic_layer(pic).view([bacth_size, cnn_kernal_size, -1])
        pic=torch.transpose(pic,1,2)
        serial,h=self.cnn_serial(pic)
        return h
class Landmark_diff_module(nn.Module):
    def __init__(self,):
        super(Landmark_diff_module,self).__init__()
        self.shake_adapt=nn.Sequential(nn.Linear(landmark_init,landmark_init),)
        self.to_memory=nn.Sequential(nn.Linear(landmark_init,landmark_init),
                                     nn.Sigmoid(),
                                     nn.Linear(landmark_init, landmark_init),
                                     nn.BatchNorm1d(landmark_init),
                                     nn.Sigmoid(),
                                     nn.Linear(landmark_init,aspect_dim),
                                     nn.Tanh())
    def forward(self,diff):
        adapt=self.shake_adapt(diff)
        re_diff=diff+adapt
        # temproal_feature,h=self.lstm_diff(re_diff)
        temproal_landmark_memory=self.to_memory(re_diff)
        temproal_landmark_memory = temproal_landmark_memory.view( 1,temproal_landmark_memory.size(0), -1)
        return temproal_landmark_memory
class Aspect_Query(nn.Module):
    def __init__(self):
        super(Aspect_Query,self).__init__()
        self.aspects_Embendding=nn.Embedding(aspect_number,aspect_dim)
        self.au_Embendding = nn.Embedding(8, aspect_dim)
        #如果用sequential包起来仅仅一个rnn，就会被报只能输入一个张量的错误
        self.query1 = nn.GRU(aspect_dim, aspect_dim, num_layers=1, batch_first=True)
        self.query2 = nn.GRU(aspect_dim, aspect_dim, num_layers=1, batch_first=True)
        self.query_for_au = nn.GRU(aspect_dim, aspect_dim, num_layers=1, batch_first=True)
        #关于西安bn还是先激活：实践中后BN，原始论文中先BN，本人先BN，
        self.project2class_with_summmary=nn.Sequential(nn.BatchNorm1d(aspect_dim),
                                                       nn.Linear(aspect_dim,aspect_dim//4),
                                                       nn.BatchNorm1d(aspect_dim // 4),
                                                       nn.Sigmoid(),
                                                       nn.Linear(aspect_dim//4,aspect_dim//16),
                                                       nn.BatchNorm1d(aspect_dim // 16),
                                                       nn.Sigmoid(),
                                                       nn.Linear(aspect_dim//16,2),
                                                       nn.Softmax())
    def forward(self,cnn_memory,landmark_memory):
        aspect_embedding=self.aspects_Embendding(torch.LongTensor([list(range(aspect_number))]*bacth_size).cuda())
        au_embedding=self.au_Embendding(torch.LongTensor([list(range(8))]*bacth_size).cuda())
        final_aspect1,h1=self.query1(aspect_embedding,cnn_memory)
        final_aspect2,h2=self.query2(final_aspect1,landmark_memory)


        transposed_final_aspect=torch.transpose(final_aspect2,1,2)
        alpha=torch.softmax(torch.bmm(final_aspect2,transposed_final_aspect),dim=-1)
        self_attention=torch.sum(torch.bmm(alpha,final_aspect2),dim=-2).view(1,bacth_size,-1)
        au_query,h=self.query_for_au(au_embedding,self_attention)
        out=self.project2class_with_summmary(au_query.contiguous().view(bacth_size*8,-1))
        return out
















# class classifier_Semantic_correlation(nn.Module):
#     def __init__(self):
#         # out_put, attention_vector_a, max_pool_, temp_landmark
#         super(classifier_Semantic_correlation,self).__init__()
#         self.feature_dim=cnn_kernal_size+cnn_kernal_size
#         self.toseq_image=nn.GRU(cnn_kernal_size,cnn_kernal_size,batch_first=True,bidirectional=False,num_layers=1)
#         self.toseq_landmark = nn.GRU(cnn_kernal_size, cnn_kernal_size, batch_first=True, bidirectional=False, num_layers=1)
#         self.Label_Embdding=nn.Embedding(8,cnn_kernal_size,)
#         self.predictor=nn.Sequential(nn.Linear(cnn_kernal_size,64),
#                                      nn.BatchNorm1d(64),
#                                      nn.ELU(),
#                                      nn.Linear(64,16),
#                                      nn.ELU(),
#                                      nn.Linear(16,2),
#                                      nn.Softmax(dim=-1)
#                                      )
#         self.landmark_filter = nn.Sequential(nn.Linear(landmark_init,landmark_init//2),
#                                              nn.ELU(),
#                                              nn.Linear(landmark_init//2, landmark_init//2),
#                                              nn.ELU(),
#                                              nn.Linear(landmark_init//2,landmark_final),
#                                              nn.ELU(),
#                                              nn.Linear(landmark_final, landmark_final),
#                                              nn.BatchNorm1d(landmark_final),
#                                              nn.Sigmoid()
#                                              )
#         self.attention_transfer=nn.Sequential(nn.Linear(cnn_kernal_size,cnn_kernal_size),
#                                               nn.Sigmoid(),
#                                               nn.Linear(cnn_kernal_size,cnn_kernal_size),
#                                               nn.Sigmoid())
#     def forward(self,h_global_correlation_feature,cnn_hot_map,label_seq,temp_landmark):
#         emb_label_seq=self.Label_Embdding(label_seq)
#         out_put,h_global_correlation_feature_=self.toseq_image(emb_label_seq,h_global_correlation_feature)
#
#         temp_landmark_hidden_state= self.landmark_filter(temp_landmark)
#         temp_landmark_hidden_state = temp_landmark_hidden_state.unsqueeze(dim=0)
#         out_put, temp_landmark_hidden_state=self.toseq_landmark(out_put, temp_landmark_hidden_state)
#
#         emb_label_seq_a=self.attention_transfer(emb_label_seq)
#         viewed_cnn_hot_map=cnn_hot_map.contiguous().view([bacth_size,cnn_kernal_size,-1])
#         viewed_cnn_hot_map_t=torch.transpose(viewed_cnn_hot_map,1,2)
#         attention_weight =torch.softmax(torch.bmm(emb_label_seq_a, viewed_cnn_hot_map),dim=2)
#         attention_vector_a=torch.bmm(attention_weight,viewed_cnn_hot_map_t)
#
#         # out_put = 0.5*out_put+0.5*attention_vector_a
#         out_put = out_put+attention_vector_a
#         out_put = out_put.view(out_put.size()[0] * 8, -1)
#         out_put_p=self.predictor(out_put)
#
#         return out_put_p
#
#
#
#
#
