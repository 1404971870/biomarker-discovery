import argparse
import os
import sys
from accelerate import Accelerator

os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7"
import shutil
import time
import datetime
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from datasets import *
import torch.nn as nn
import torch.nn.functional as F
import torch
torch.autograd.set_detect_anomaly(True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    parser.add_argument("--save_dir", type=str, default="checkpoints_concat_normal", help="folder to save checkpoints")
    parser.add_argument('--train_data_path', type=str, default="/data/sunys/program/PyTorch-GAN-master/data/8474_concat_normal_msi_images", help='Path to the training data directory.')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pth.tar', help='Path to the checkpoint file.')
    parser.add_argument('--save_image_path', type=str, default="images_concat_normal", help='Path to the training data directory.')

    opt = parser.parse_args()
    print(opt)
    os.makedirs(opt.save_dir, exist_ok=True)
    os.makedirs(opt.save_image_path, exist_ok=True)

    accelerator = Accelerator()

    img_shape = (opt.channels, opt.img_size, opt.img_size)

    # cuda = True if torch.cuda.is_available() else False
    device = accelerator.device

    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)


    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()

            self.init_size = opt.img_size // 4
            self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

            self.conv_blocks = nn.Sequential(
                nn.InstanceNorm2d(128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.InstanceNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.InstanceNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
                nn.Tanh(),
            )

        def forward(self, z):
            out = self.l1(z)
            out = out.view(out.shape[0], 128, self.init_size, self.init_size)
            img = self.conv_blocks(out)
            return img


    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()

            def discriminator_block(in_filters, out_filters, bn=True):
                block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=False), nn.Dropout2d(0.25)]
                if bn:
                    block.append(nn.InstanceNorm2d(out_filters, 0.8))
                return block

            self.model = nn.Sequential(
                *discriminator_block(opt.channels, 16, bn=False),
                *discriminator_block(16, 32),
                *discriminator_block(32, 64),
                *discriminator_block(64, 128),
            )

            # The height and width of downsampled image
            ds_size = opt.img_size // 2 ** 4
            # ds_size ** 2 是特征图的空间维度（宽度和高度）,就是长乘宽
            self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

        def forward(self, img):
            out = self.model(img)
            out = out.view(out.shape[0], -1)
            validity = self.adv_layer(out)

            return validity


    # Loss function
    # 二元交叉熵损失函数（Binary Cross-Entropy Loss）
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    # if cuda:
    #     generator.cuda()
    #     discriminator.cuda()
    #     adversarial_loss.cuda()

    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)
    # Initialize weights
    # 如果用了checkpoint就先别初始化权重了
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    def get_train_loader(batch_size, train_data_path):
        transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像转换为Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
        ])
        train_dataset = customDataset(root=train_data_path, transforms=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # # Debug: Print a batch of data
        # data_iter = iter(train_loader)
        # images, labels = next(data_iter)
        # print(f'Train images batch shape: {images.shape}')
        # print(f'Train labels batch shape: {labels.shape}')

        return train_loader

    def save_checkpoint(state, is_best, save):
        filename = os.path.join(save, 'checkpoint.pth.tar')
        torch.save(state, filename)
        if is_best:
            best_filename = os.path.join(save, 'best_model.pth.tar')
            shutil.copyfile(filename, best_filename)

    train_data_path = opt.train_data_path
    dataloader = get_train_loader(opt.batch_size, train_data_path)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    accelerator.print(f'device {str(accelerator.device)} is used!')

    # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    generator, discriminator, optimizer_G, optimizer_D, dataloader = accelerator.prepare(
        generator, discriminator, optimizer_G, optimizer_D, dataloader
    )



    # ----------
    #  Training
    # ----------

    # 定义checkpoint路径
    checkpoint_path = os.path.join(opt.save_dir, opt.checkpoint_path)

    # 尝试加载checkpoint
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from {checkpoint_path}, starting from epoch {start_epoch}.")
    else:
        print("No checkpoint found, starting training from scratch.")
        start_epoch = 0

    prev_time = time.time()

    for epoch in range(start_epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            imgs = batch["A"]
            # Adversarial ground truths
            valid = torch.ones((imgs.shape[0], 1)).requires_grad_(False).to(device)
            fake = torch.zeros((imgs.shape[0], 1)).requires_grad_(False).to(device)

            # Configure input
            real_imgs = imgs.to(device, dtype=torch.float32)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z_np = np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))
            z = torch.from_numpy(z_np).float().to(device)
            # z = torch.from_numpy(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))).float().to(device)

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            # 这行代码计算生成器的对抗性损失 g_loss。adversarial_loss 函数比较了判别器对假图像的判断和真实标签 valid，计算损失值。
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            # g_loss.backward()
            accelerator.backward(g_loss)

            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            # 判断判别器能不能正确把真实图片判断成真实的loss
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            # 判断判别器能不能正确把虚假图片判断成虚假的loss
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

            d_loss = (real_loss + fake_loss) / 2

            # d_loss.backward()
            accelerator.backward(d_loss)
            optimizer_D.step()
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            accelerator.print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), time_left)
            )
            if accelerator.is_local_main_process:
                if batches_done % opt.sample_interval == 0:
                    save_image(gen_imgs.data[:25], "%s/%d.png" % (opt.save_image_path, batches_done), nrow=5, normalize=True)

        # 保存周期性模型
        if (epoch + 1)  == opt.n_epochs:
            accelerator.wait_for_everyone()
            unwrapped_generator = accelerator.unwrap_model(generator)
            unwrapped_discriminator = accelerator.unwrap_model(discriminator)
            accelerator.save_model(unwrapped_generator, opt.save_dir + "_generator",safe_serialization=False)
            accelerator.save_model(unwrapped_discriminator, opt.save_dir+ "_discriminator", safe_serialization=False)

            # print('Saving model...')
            # checkpoint = {
            #     'epoch': epoch + 1,
            #     'generator_state_dict': generator.state_dict(),
            #     'discriminator_state_dict': discriminator.state_dict(),
            #     'optimizer_G_state_dict': optimizer_G.state_dict(),
            #     'optimizer_D_state_dict': optimizer_D.state_dict(),
            # }
            #
            # torch.save(checkpoint, os.path.join(opt.save_dir, f'checkpoint_{epoch + 1}.pth.tar'))

if __name__ == "__main__":
    main()