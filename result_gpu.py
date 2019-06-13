import scipy.io as sio
from scipy import signal
import torch
from torch import tensor,nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import datasets,transforms
from PIL import Image
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
import os
import numpy as np
from scipy import signal
import pywt

#传入完整信号，返回保存波峰list
def extPeak(signal):
    flag=0
    temp=signal[1]
    top=[]
    maxpoint=max(temp)
    thre=maxpoint/4*2.6
    if maxpoint<0.3:
        thre=0.15
    for n in range(5000):
        if temp[n]>thre:
            if len(top)==0:
                top.append(n)
            else:
                if (n-top[len(top)-1])>50:
                    top.append(n)
                elif temp[n]>temp[top[len(top)-1]]:
                    top[len(top)-1]=n
    if len(top)<5:
        temp=signal[3]
        top=[]
    for n in range(5000):
        if temp[n]>thre:
            if len(top)==0:
                top.append(n)
            else:
                if (n-top[len(top)-1])>50:
                    top.append(n)
                elif temp[n]>temp[top[len(top)-1]]:
                    top[len(top)-1]=n
    return top

#传入某一行的心电信号
#注意python和matlab不一样，python包头不包尾
def extractFeawave(rawecg,top=[]):
    b,a=signal.butter(6,[0.008,0.4],'bandpass')
    ecg=signal.filtfilt(b,a,rawecg)
    if len(top)>7:
        fea1=ecg[top[4]-79:top[4]+1121] 
        fea2=ecg[top[4]-149:top[4]+1050]
        fea3=ecg[top[5]-149:top[5]+1050]
        fea4=ecg[top[3]-149:top[3]+1050]
        fea5=ecg[top[5]-79:top[5]+1120]
        fea6=ecg[top[3]-79:top[3]+1120]
        fea7=ecg[top[3]+1:top[3]+1200]     
        fea8=ecg[top[5]+1:top[5]+1200] 
        fea9=ecg[top[4]+1:top[4]+1200]
    else:
        fea1=ecg[501:1700] 
        fea2=ecg[901:2100]
        fea3=ecg[1301:2500]
        fea4=ecg[1701:2900]
        fea5=ecg[2101:3300]
        fea6=ecg[2501:3700]
        fea7=ecg[2901:4100]     
        fea8=ecg[3401:4600] 
        fea9=ecg[3801:5000]        	
    return fea1,fea2,fea3,fea4,fea5,fea6,fea7,fea8,fea9

#传入特征波，直接返回转换后np格式的图片
def ecgcwt(feawave):
    #transform
    tran=transforms.Resize((448,448))
    coef, freqs=pywt.cwt(feawave,np.arange(5,250),'gaus4')
    im=Image.fromarray(coef)
    im=tran(im)
    pic=np.array(im)
    return pic

#模型
class ResidualBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1,shortcut=None):
        super(ResidualBlock,self).__init__()
        self.left=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right=shortcut
        
    def forward(self,x):
        out=self.left(x)
        residual=x if self.right is None else self.right(x)
        out+=residual
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self,num_classes=2):
        super(ResNet,self).__init__()
        self.pre1=nn.Conv2d(1,64,7,2,3,bias=False)
        self.pre2=nn.BatchNorm2d(64)
        self.pre3=nn.ReLU(inplace=True)
        self.pre4=nn.MaxPool2d(3,2,1)
        self.layer1=self._make_layer(64,64,3)
        self.layer2=self._make_layer(64,128,4,stride=2)
        self.layer3=self._make_layer(128,256,6,stride=2)
        self.layer4=self._make_layer(256,512,3,stride=2)
        self.fc1=nn.Linear(512,80)
        self.fc2=nn.Linear(80,num_classes)

    def _make_layer(self,inchannel,outchannel,bloch_num,stride=1):

        shortcut=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers=[]
        layers.append(ResidualBlock(inchannel,outchannel,stride,shortcut))
        for i in range(1,bloch_num):
            layers.append(ResidualBlock(outchannel,outchannel))
        return nn.Sequential(*layers)

    def forward(self,x):
        x=self.pre1(x)
        x=self.pre2(x)
        x=self.pre3(x)
        x=self.pre4(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=F.avg_pool2d(x,14)
        x=x.view(x.size(0),-1)
        out1=self.fc1(x);
        x=F.dropout(out1, p=0.5, training=self.training)
        x=self.fc2(x);
        return F.dropout(x, p=0.5, training=self.training)

#加载模型
net_a=torch.load('./Zhang-s-Sisters_GPU/model_a_gpu.pkl')
net_b=torch.load('./Zhang-s-Sisters_GPU/model_b_gpu.pkl')
net_c=torch.load('./Zhang-s-Sisters_GPU/model_c_gpu.pkl')
net_d=torch.load('./Zhang-s-Sisters_GPU/model_d_gpu.pkl')
net_e=torch.load('./Zhang-s-Sisters_GPU/model_e_gpu.pkl')
net_f=torch.load('./Zhang-s-Sisters_GPU/model_f_gpu.pkl')
net_g=torch.load('./Zhang-s-Sisters_GPU/model_g_gpu.pkl')
net_h=torch.load('./Zhang-s-Sisters_GPU/model_h_gpu.pkl')
net_i=torch.load('./Zhang-s-Sisters_GPU/model_i_gpu.pkl')
net_j=torch.load('./Zhang-s-Sisters_GPU/model_j_gpu.pkl')
net_k=torch.load('./Zhang-s-Sisters_GPU/model_k_gpu.pkl')
net_l=torch.load('./Zhang-s-Sisters_GPU/model_l_gpu.pkl')

#transform
tran=transforms.Resize((448,448))
log=open('answers.txt','w')

pathDir =  os.listdir('./TEST/')
for childdir in pathDir:  
    filepath = os.path.join('%s/%s' % ('./TEST/', childdir))#合成文件名  
    name=childdir[0:7]
    log.write(name)
    log.write('\t')
    print(name)
    #读取文件
    top=[]
    data=sio.loadmat(file_name=filepath)
    ecg=data['data']#可获得numpy类型的信号
    top=extPeak(ecg)
    cls=0

    #对每一个特征波分开处理
    for i in range(12):
        sig=ecg[i]
        testdata=[]
        tempflag=0
        fea_1,fea_2,fea_3,fea_4,fea_5,fea_6,fea_7,fea_8,fea_9=extractFeawave(sig,top)
        if i==0:
            temp=ecgcwt(fea_1)
            testdata.append(temp)
            testdata=np.array([np.array(testdata)])
            temp=ecgcwt(fea_2)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_3)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_4)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_5)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_6)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_7)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_8)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_9)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            testdata=torch.from_numpy(testdata).type(torch.FloatTensor)
            testdata=Variable(testdata).cuda()
            test_output= net_a(testdata) 
            pred_y = torch.max(test_output.cpu(),1)[1].data.numpy().squeeze()
            print(pred_y)
        elif i==1:
            temp=ecgcwt(fea_1)
            testdata.append(temp)
            testdata=np.array([np.array(testdata)])
            temp=ecgcwt(fea_2)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_3)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_4)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_5)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_6)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_7)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_8)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_9)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            testdata=torch.from_numpy(testdata).type(torch.FloatTensor)
            testdata=Variable(testdata).cuda()
            test_output= net_b(testdata) 
            pred_y = torch.max(test_output.cpu(),1)[1].data.numpy().squeeze()
            print(pred_y)
        elif i==2:
            temp=ecgcwt(fea_1)
            testdata.append(temp)
            testdata=np.array([np.array(testdata)])
            temp=ecgcwt(fea_2)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_3)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_4)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_5)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_6)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_7)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_8)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_9)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            testdata=torch.from_numpy(testdata).type(torch.FloatTensor)
            testdata=Variable(testdata).cuda()
            test_output= net_c(testdata) 
            pred_y = torch.max(test_output.cpu(),1)[1].data.numpy().squeeze()
            print(pred_y)
        elif i==3:
            temp=ecgcwt(fea_1)
            testdata.append(temp)
            testdata=np.array([np.array(testdata)])
            temp=ecgcwt(fea_2)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_3)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_4)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_5)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_6)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_7)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_8)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_9)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            testdata=torch.from_numpy(testdata).type(torch.FloatTensor)
            testdata=Variable(testdata).cuda()
            test_output= net_d(testdata) 
            pred_y = torch.max(test_output.cpu(),1)[1].data.numpy().squeeze()
            print(pred_y)
        elif i==4:
            temp=ecgcwt(fea_1)
            testdata.append(temp)
            testdata=np.array([np.array(testdata)])
            temp=ecgcwt(fea_2)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_3)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_4)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_5)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_6)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_7)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_8)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_9)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            testdata=torch.from_numpy(testdata).type(torch.FloatTensor)
            testdata=Variable(testdata).cuda()
            test_output= net_e(testdata) 
            pred_y = torch.max(test_output.cpu(),1)[1].data.numpy().squeeze()
            print(pred_y)
        elif i==5:
            temp=ecgcwt(fea_1)
            testdata.append(temp)
            testdata=np.array([np.array(testdata)])
            temp=ecgcwt(fea_2)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_3)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_4)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_5)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_6)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_7)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_8)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_9)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            testdata=torch.from_numpy(testdata).type(torch.FloatTensor)
            testdata=Variable(testdata).cuda()
            test_output= net_f(testdata) 
            pred_y = torch.max(test_output.cpu(),1)[1].data.numpy().squeeze()
            print(pred_y)
        elif i==6:
            temp=ecgcwt(fea_1)
            testdata.append(temp)
            testdata=np.array([np.array(testdata)])
            temp=ecgcwt(fea_2)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_3)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_4)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_5)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_6)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_7)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_8)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_9)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            testdata=torch.from_numpy(testdata).type(torch.FloatTensor)
            testdata=Variable(testdata).cuda()
            test_output= net_g(testdata) 
            pred_y = torch.max(test_output.cpu(),1)[1].data.numpy().squeeze()
            print(pred_y)
        elif i==7:
            temp=ecgcwt(fea_1)
            testdata.append(temp)
            testdata=np.array([np.array(testdata)])
            temp=ecgcwt(fea_2)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_3)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_4)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_5)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_6)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_7)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_8)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_9)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            testdata=torch.from_numpy(testdata).type(torch.FloatTensor)
            testdata=Variable(testdata).cuda()
            test_output= net_h(testdata) 
            pred_y = torch.max(test_output.cpu(),1)[1].data.numpy().squeeze()
            print(pred_y)
        elif i==8:
            temp=ecgcwt(fea_1)
            testdata.append(temp)
            testdata=np.array([np.array(testdata)])
            temp=ecgcwt(fea_2)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_3)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_4)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_5)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_6)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_7)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_8)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_9)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            testdata=torch.from_numpy(testdata).type(torch.FloatTensor)
            testdata=Variable(testdata).cuda()
            test_output= net_i(testdata) 
            pred_y = torch.max(test_output.cpu(),1)[1].data.numpy().squeeze()
            print(pred_y)
        elif i==9:
            temp=ecgcwt(fea_1)
            testdata.append(temp)
            testdata=np.array([np.array(testdata)])
            temp=ecgcwt(fea_2)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_3)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_4)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_5)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_6)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_7)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_8)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_9)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            testdata=torch.from_numpy(testdata).type(torch.FloatTensor)
            testdata=Variable(testdata).cuda()
            test_output= net_j(testdata) 
            pred_y = torch.max(test_output.cpu(),1)[1].data.numpy().squeeze()
            print(pred_y)
        elif i==10:
            temp=ecgcwt(fea_1)
            testdata.append(temp)
            testdata=np.array([np.array(testdata)])
            temp=ecgcwt(fea_2)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_3)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_4)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_5)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_6)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_7)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_8)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_9)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            testdata=torch.from_numpy(testdata).type(torch.FloatTensor)
            testdata=Variable(testdata).cuda()
            test_output= net_k(testdata) 
            pred_y = torch.max(test_output.cpu(),1)[1].data.numpy().squeeze()
            print(pred_y)
        else:
            temp=ecgcwt(fea_1)
            testdata.append(temp)
            testdata=np.array([np.array(testdata)])
            temp=ecgcwt(fea_2)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_3)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_4)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_5)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_6)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_7)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_8)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            temp=ecgcwt(fea_9)
            testdata=np.concatenate((testdata,np.array([np.array([temp])])),axis=0)
            testdata=torch.from_numpy(testdata).type(torch.FloatTensor)
            testdata=Variable(testdata).cuda()
            test_output= net_l(testdata) 
            pred_y = torch.max(test_output.cpu(),1)[1].data.numpy().squeeze()
            print(pred_y)
        sum=np.sum(pred_y)
        print(sum)
        if sum>5:
            cls=cls+1
    
        print(cls)

    if cls>5:
        finalcls=1
    else:
        finalcls=0
  
    print(finalcls)
    log.write(str(finalcls))
    log.write('\n')   
    
log.close()