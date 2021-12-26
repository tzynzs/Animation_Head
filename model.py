import torch.nn as nn


class Generator(nn.Module):
    def __init__(self,opt):
        super(Generator,self).__init__()
        ngf=opt.generator_feature
        nz=opt.noise_size
        self.main=nn.Sequential(
            nn.ConvTranspose2d(nz,ngf*8,kernel_size=4,stride=1,padding=0,bias=False),   #输入1*1*nz
            nn.BatchNorm2d(ngf*8),    #归一化处理
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8,ngf*4, kernel_size=4, stride=2, padding=1, bias=False),    #输入4*4*ngf*8
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1,bias=False),    #输入8*8*ngf*4
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf* 2, ngf, kernel_size=4, stride=2, padding=1,bias=False),     #输入16*16*ngf*2
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, kernel_size=5, stride=3, padding=1,bias=False),      #输入32*32*ngf
            nn.Tanh()       #输出96*96*3
        )

    def forward(self,x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self,opt):
        super(Discriminator,self).__init__()
        ndf=opt.discriminator_feature
        self.main=nn.Sequential(
            nn.Conv2d(3,ndf,kernel_size=5,stride=3,padding=1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),     #防止Relu产生梯度消失问题
            nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf*2,ndf* 4, kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf*4,ndf * 8, kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf*8, 1, kernel_size=4,stride=1,padding=0,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.main(x).view(-1)
