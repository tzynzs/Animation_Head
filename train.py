import matplotlib.pyplot as plt
import torch
import tqdm
import torchvision as tv
from model import Generator,Discriminator
from config import opt

plt.rcParams['font.sans-serif'] = ['KaiTi']

d_every = 1
g_every = 5

transform=tv.transforms.Compose([
    tv.transforms.Resize(96),
    tv.transforms.CenterCrop(96),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

dataset=tv.datasets.ImageFolder(root=opt.data_path,transform=transform)
dataloader=torch.utils.data.DataLoader(dataset,batch_size=opt.batch_size,shuffle=True,num_workers=4,drop_last=True)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

G = Generator(opt).to(device)
D = Discriminator(opt).to(device)


optimizer_g = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = torch.nn.BCELoss()

true_lables = torch.ones(opt.batch_size).to(device)
fake_lables = torch.zeros(opt.batch_size).to(device)
noises = torch.randn(opt.batch_size, opt.noise_size, 1, 1).to(device)  #正态分布取样，noises为4维张量，共batch_size组，每组noise_size个1行1列的张量

def train():

    for i,(image,_) in tqdm.tqdm(enumerate(dataloader)):    # type((image,_)) = <class 'list'>, len((image,_)) = 2 * 256 * 3 * 96 * 96
        real_image=image.to(device)
        if (i + 1) % d_every == 0:

            optimizer_d.zero_grad()
            output=D(real_image)
            d_real_loss=criterion(output,true_lables)
            d_real_loss.backward()

            noises.data.copy_(torch.randn(opt.batch_size, opt.noise_size, 1, 1))
            fake_img=G(noises).detach()
            fake_output=D(fake_img)
            d_fake_loss=criterion(fake_output,fake_lables)
            d_fake_loss.backward()
            optimizer_d.step()      #每个batch训练更新参数空间

        if (i + 1) % g_every == 0:

            optimizer_g.zero_grad()
            noises.data.copy_(torch.randn(opt.batch_size, opt.noise_size, 1, 1))
            fake_img=G(noises)
            fake_output=D(fake_img)
            g_loss=criterion(fake_output,true_lables)
            g_loss.backward()
            optimizer_g.step()

        if (i+1)%5==0:
            print('d_real_loss:{}'.format(d_real_loss),'d_fake_loss:{}'.format(d_fake_loss),'g_loss:{}'.format(g_loss))

def show(num):
    fix_fake_imgs=G(noises).data.cpu()[:64]*0.5+0.5
    fig=plt.figure(1)

    i=1
    for img in fix_fake_imgs:
        fig.add_subplot(8,8,eval('%d' % i))
        plt.axis(False)
        plt.imshow(img.permute(1,2,0))
        i+=1

    plt.subplots_adjust(None,None,None,None,0.1,0.1)
    plt.suptitle("第%d次迭代结果" %num)
    plt.savefig(opt.path.join(opt.FAKE_FACES_DIR, str(num) + '.png'))


if __name__=='__main__':
    for i in range(100):
        print('第{}次迭代'.format(i+1))
        train()
        if(((i+1)%10==0)|(i+1==1)):
            show(i+1)