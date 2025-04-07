import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import time
import sys

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models_single_train import *
from datasets import *

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--dataset_root", type=str, default="/data/sunys/program/benchmark_VAE-main/my_space/data/multi_model", help="name of the dataset")
parser.add_argument("--dataset_name", type=str, default="1000epch_single_train_WSIimage_normal", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--pretrain_model", type=str, default="mwsi_pretrain_normal_1000", help="name of the dataset")

parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

def main():
    os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
    os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

    cuda = True if torch.cuda.is_available() else False

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()

    # Loss weight of L1 pixel-wise loss between translated image and real image
    lambda_pixel = 100

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

    # Configure dataloaders
    transforms_msi = [
        transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    transforms_wsi = [
        transforms.Resize((2048, 2048), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    # 使用ImageLoader类的示例，读取msi数据为一个332个通道矩阵
    # MSI = MSI_dataset('/data/sunys/program/PyTorch-GAN-master/data/12/')
    # msi_filename = MSI.get_image_names()
    # MSI_image = MSI.create_image_tensor()
    # MSI_image = MSI_image.permute(1, 0, 2, 3)
    #
    #
    low_wsi = load_low_wsi('/data/sunys/program/PyTorch-GAN-master/data/normal_nobackground_21C.png')
    # high_wsi = load_high_wsi('/data/sunys/program/benchmark_VAE-main/my_space/data/multi_model/train/wsi_high/padding_rgb_nobackground_23m_1240_2136.png')

    # slide_embedding = load_and_reshape_tensor('/data/sunys/program/prov-gigapath-main/data/normal_last_layer_embeddings.pth')

    dataloader = DataLoader(
        high_customDataset(root_msi='/data/sunys/program/PyTorch-GAN-master/data/8474_nomal_valid_3',
                      transforms_msi=transforms_msi, transforms_wsi=transforms_wsi, mode="train"),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    # val_dataloader = DataLoader(
    #     customDataset(root=opt.dataset_root, transforms_msi=transforms_msi,transforms_wsi=transforms_wsi, mode="val"),
    #     batch_size=10,
    #     shuffle=True,
    #     num_workers=1,
    # )

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # generator, discriminator, optimizer_G, optimizer_D, dataloader = accelerator.prepare(
    #     generator, discriminator, optimizer_G, optimizer_G, dataloader)
    # ----------
    #  Training
    # ----------

    # 记录开始时间
    start_time = time.time()

    # 记录总的batch数量
    total_batches = len(dataloader)

    # 应该在这里循环加载权重，然后对dataloader里的一个数据训练20轮后，又重新加载权重，然后加载下一个数据集，训练20抡
    # 如果放在    for i, batch in enumerate(dataloader):里的话，每次训练一个数据就重新加载权重了
    # 如果放在for epoch in range(opt.epoch, opt.n_epochs):里的话，每次训练一轮就重新加载权重了
    # 应该先一个for循环，加载数据，然后加载权重，再训练；然后再加载数据
    # 注意enumerate返回值有两个,一个是序号，一个是数据（包含训练数据和标签）
    for i, batch in enumerate(dataloader):
        # se = Variable(slide_embedding.type(Tensor))

        # 这一步把msi转移到tensor上占用了893mb的显存
        real_low_A = Variable(low_wsi.type(Tensor))
        real_B = Variable(batch["A"].type(Tensor))
        valid = Variable(Tensor(np.ones((real_low_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_low_A.size(0), *patch))), requires_grad=False)
        # ———————————重新加载模型权重————————————————
        # Initialize generator and discriminator
        generator = GeneratorUNet()
        discriminator = Discriminator()

        if cuda:
            generator = generator.cuda()
            discriminator = discriminator.cuda()
            criterion_GAN.cuda()
            criterion_pixelwise.cuda()

        if opt.epoch != 0:
            # Load pretrained models
            generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.pretrain_model, opt.epoch)))
            discriminator.load_state_dict(
                torch.load("saved_models/%s/discriminator_%d.pth" % (opt.pretrain_model, opt.epoch)))
        else:
            # Initialize weights
            generator.apply(weights_init_normal)
            discriminator.apply(weights_init_normal)

        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        for epoch in range(opt.epoch, opt.n_epochs):
            # ------------------
            #  Train Generators
            # ------------------
            optimizer_G.zero_grad()

            # GAN loss
            # 这里fake_b是(1,332,256,256)
            fake_B = generator(real_low_A)

            pred_fake = discriminator(fake_B, real_low_A)
            # 在训练生成器时，我们希望判别器将生成的数据误判为真实数据，因此生成器的损失是当判别器预测为真实时的损失。
            loss_GAN = criterion_GAN(pred_fake, valid)
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)

            # Total loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel

            loss_G.backward()

            # accelerator.backward(loss_G)

            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(real_B, real_low_A)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_B.detach(), real_low_A)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()

            # accelerator.backward(loss_D)

            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------
            # 每个epoch结束后计算时间
            # 当前时间


            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d][D loss: %f] [G loss: %f, pixel: %f, adv: %f]"
                % (
                    epoch,
                    opt.n_epochs,
                    loss_D.item(),
                    loss_G.item(),
                    loss_pixel.item(),
                    loss_GAN.item(),
                )
            )

            # If at sample interval save image
            if epoch == opt.n_epochs - 1:
                fake_B = fake_B.squeeze(0).cpu()
                # 确保保存模型的目录存在
                save_fakeb_dir = "images/%s" % (opt.dataset_name)
                os.makedirs(save_fakeb_dir, exist_ok=True)
                image = transforms.ToPILImage()(fake_B)
                filename = batch["filename"]
                filename = filename[0]
                filename += '.png'
                image.save(os.path.join(save_fakeb_dir, filename))

        # 计算已经过去的时间和当前进度
        elapsed_time = time.time() - start_time
        batches_done = i + 1  # 完成的batch数量，包括当前这个
        batches_left = total_batches - batches_done  # 剩余的batch数量

        # 如果还有剩余的batch，计算剩余时间
        if batches_left > 0:
            average_time_per_batch = elapsed_time / batches_done
            remaining_time = average_time_per_batch * batches_left
        else:
            remaining_time = 0

        # 将剩余时间转换为小时和分钟
        hours = int(remaining_time // 3600)
        minutes = int((remaining_time % 3600) // 60)
        seconds = int(remaining_time % 60)

        # 打印当前进度和剩余时间
        print(
            f"Batch {batches_done}/{total_batches}, Elapsed Time: {elapsed_time:.2f}s, Remaining Time: {hours}h {minutes}m {seconds}s")
            # 等待每个GPU上的模型执行完当前的epoch，并进行合并同步
            # accelerator.wait_for_everyone()
            # generator = accelerator.unwrap_model(generator)
            # discriminator = accelerator.unwrap_model(discriminator)
            # accelerator.save(generator,"saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
            # accelerator.save(discriminator,"saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))

        # if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        #     # Save model checkpoints
        #     accelerator.save(generator,"saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
        #     accelerator.save(discriminator,"saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))

if __name__ == "__main__":
    main()