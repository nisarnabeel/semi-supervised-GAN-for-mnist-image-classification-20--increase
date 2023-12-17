# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 22:58:51 2023

@author: nisar
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:33:42 2023

@author: nisar
"""

import argparse
import os
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

import torch.nn as nn
import torch.nn.functional as F
import torch
import itertools
from tqdm import tqdm
from sklearn import metrics

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--num_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.linear=nn.Linear(latent_dim, 256 * 7* 7)
        self.model = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=(1,1),output_padding=(1,1)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=(1,1)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 1, 3, stride=2, padding=(1,1),output_padding=(1,1)),
            nn.Tanh()
        )

    def forward(self, z):
        out=self.linear(z)
        out=out.view(-1,256,7,7)
        return self.model(out)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(2048, 10)  # Assuming input image size is 28x28
        )
        # Output layers
        #self.adv_layer = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())

    def forward(self, img):
        label = self.model(img)
        #validity = self.adv_layer(label)
        z_x = torch.sum(torch.exp(label), dim=-1, keepdim=True)
        d_x = z_x / (z_x + 1)

        return d_x ,label


torch.manual_seed(42)

# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
os.makedirs("data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=1,
    shuffle=True,
)
num_samples_to_select = 100
selected_samples = []
selected_labels=[]
im=[]
la=[]
print(len(dataloader))

labeled_imgs = []
labeled_labels = []
unlabeled_imgs = []
unlabeled_labels = []

# Initialize a counter for each class
class_counter = [0] * 10

# Iterate through the dataset
for images, labels in dataloader:
    # Extract the class label
    label = int(labels)

    # Check if the class has less than ten samples already
    s=torch.rand(1)
    if (s>=0.5 and class_counter[label] < 10):
        labeled_imgs.append(images)
        labeled_labels.append(labels)
        class_counter[label] += 1
    else:
        unlabeled_imgs.append(images)
        unlabeled_labels.append(labels)

# Concatenate the lists to create labeled and unlabeled datasets
labeled_dataset = torch.utils.data.TensorDataset(torch.cat(labeled_imgs), torch.cat(labeled_labels))
# unlabeled_dataset = torch.utils.data.TensorDataset(torch.cat(unlabeled_imgs), torch.cat(unlabeled_labels))

print(f"Number of labeled samples: {len(labeled_dataset)}")
# print(f"Number of unlabeled samples: {len(unlabeled_dataset)}")



dataloader = DataLoader(dataset=labeled_dataset, batch_size=opt.batch_size, shuffle=True)
# train_dataset=TensorDataset(im,la)
dataloader_unlabeled = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=100,
    shuffle=True,
)

print("train data loader len is",len(dataloader.dataset))
print("unlabeled len is",len(dataloader_unlabeled.dataset))

# # Print the shape of the selected samples and labels


test_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/mnist",
        train=False,
        download=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=False,drop_last=True
)


# # Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# # ----------
# #  Training
# # ----------
im_sr=[]
im_tgt=[]
g_loss=torch.tensor(0)
for epoch in range(opt.n_epochs):
    n_batches = max(len(dataloader), len(dataloader_unlabeled))
    #print(n_batches)
    dataloader_iter = iter(dataloader)
    batches = zip(itertools.cycle(dataloader),dataloader_unlabeled)

    for (imgs,labels), (images_tgt, target_labels) in tqdm(batches, leave=False, total=n_batches):
        batch_size = imgs.shape[0]
        im_sr.append(imgs)
        im_tgt.append(images_tgt)
        

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        #valid_unsup = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)

        # fake_aux_gt = Variable(LongTensor(batch_size).fill_(opt.num_classes), requires_grad=False)
        # fake_aux_gt_uns = Variable(LongTensor(100).fill_(opt.num_classes), requires_grad=False)
        # fake_uns = Variable(FloatTensor(100, 1).fill_(0.0), requires_grad=False)



        # # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        images_tgt = Variable(images_tgt.type(FloatTensor))

        labels = Variable(labels.type(LongTensor))
        optimizer_D.zero_grad()

        # Loss for real images
        _,real_aux = discriminator(real_imgs)
       # d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 3
        d_real_loss = auxiliary_loss(real_aux, labels)


        # Loss for fake images
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_imgs = generator(z)
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = adversarial_loss(fake_pred, fake)

        # disc unsupervised
        unsuper_pred,unsuper_aux = discriminator(images_tgt)
        d_unsuper_loss = adversarial_loss(unsuper_pred, valid)
                          # + auxiliary_loss(unsuper_aux, fake_aux_gt_uns)) / 3


        # Total discriminator loss
        #d_loss = (d_real_loss + d_fake_loss)
        d_loss = (d_real_loss+d_fake_loss+d_unsuper_loss)
        d_loss.backward()
        optimizer_D.step()

        #d_loss = (d_real_loss + d_fake_loss) / 2

        #d_loss=d_real_loss
        preds=np.concatenate((fake_pred.data.cpu().numpy(),unsuper_pred.data.cpu().numpy()),axis=0)
        
        # Calculate discriminator accuracy
        # pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy(),unsuper_aux.data.cpu().numpy()], axis=0)
        # gt = np.concatenate([labels.data.cpu().numpy(), fake_aux_gt.data.cpu().numpy(),fake_aux_gt_uns.data.cpu().numpy()], axis=0)
        pred = np.concatenate([real_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy()], axis=0)
        #pred_2=np.concatenate([fake_pred.data.cpu().max(1)[1].numpy(),unsuper_pred.data.cpu().max(1)[1].numpy()],axis=0)
        gt_2=np.concatenate([fake.data.cpu().numpy(),valid.data.cpu().numpy()],axis=0)
        
       
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)
        d_acc_2 = metrics.accuracy_score(gt_2, preds>0.5)
        




        # # -----------------
        # #  Train Generator
        # # -----------------
        
        optimizer_G.zero_grad()

        # Sample noise and labels as generator input

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        validity, _ = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()



      
        print(
            "[Epoch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f] [unsup acc:%d%%]"
            % (epoch, opt.n_epochs, d_loss.item(), 100 * d_acc, g_loss.item(),100*d_acc_2)
        )

        # batches_done = epoch * len(dataloader) + i
        # if batches_done % opt.sample_interval == 0:
        #     save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
    if epoch%2==0:
        test_loss = 0
        correct_test = 0
        test_acc=0
        generator.eval()
        discriminator.eval()
        for i, (samples, labels) in enumerate(test_dataloader):
            with torch.no_grad():
                samples = samples.to('cuda')
                labels = labels.to('cuda')
                _,predict_output = discriminator(samples)
                predict_output=predict_output[:,:10]
               # predict_output=predict_output[:,:10]
                predicted_label = torch.max(predict_output, 1)[1]       
                correct_test += (predicted_label == labels).sum().item()
        test_loss /= len(test_dataloader)
        test_acc = 100 * float(correct_test) / len(test_dataloader.dataset)
        print("test acc is",test_acc)
        s=torch.concatenate(im_sr,dim=0)
        print(s.shape,torch.unique(s,dim=0).shape)
        p=torch.concatenate(im_tgt,dim=0)
        print(p.shape,torch.unique(p,dim=0).shape)