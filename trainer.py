#-*- coding:utf-8 -*-

import os
import time
import torch
import datetime
import torch.nn as nn
from torchvision.utils import save_image
from losses import PerceptualLoss, GANLoss, MultiscaleRecLoss
from utils import Logger, denorm, ImagePool
from models import Generator, Discriminator
from metrics.NIMA.CalcNIMA import calc_nima
from metrics.CalcPSNR import calc_psnr
from metrics.CalcSSIM import calc_ssim
from tqdm import *
from data_loader import InputFetcher
from pathlib import Path

from data_loader import get_train_loader, get_test_loader


class Trainer(object):
    def __init__(self, loaders, args):
        # data loader
        self.loaders = loaders
        
        # Model configuration.
        self.args = args
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"        # device="mps"

        # Directories. (save_root_dir : default='./results')
        self.model_save_path    = os.path.join(args.save_root_dir, args.version, args.model_save_path)     # default='UEGAN-FiveK'
        self.sample_path        = os.path.join(args.save_root_dir, args.version, args.sample_path)
        self.log_path           = os.path.join(args.save_root_dir, args.version, args.log_path)
        self.val_result_path    = os.path.join(args.save_root_dir, args.version, args.val_result_path)

        # duhyeonkim updated : to reuse memory, popping loss and image dictionary from def logging:
        self.loss = {}
        self.images = {}

        # Build the model and tensorboard.
        self.build_model()  # generator, discriminator, multi-GPU, weight initialization, optimizer, learning rate scheduler, image pool

        if self.args.use_tensorboard:           # default=False
            self.build_tensorboard()

    def train(self):
        """ Train UEGAN ."""
        self.fetcher = InputFetcher(self.loaders.ref)           # train
        self.fetcher_val = InputFetcher(self.loaders.val)       # val (not test)

        self.train_steps_per_epoch = len(self.loaders.ref)
        self.model_save_step = int(self.args.model_save_epoch * self.train_steps_per_epoch)         # model_save_epoch(default:1)

        # set nima, psnr, ssim global parameters
        if self.args.is_test_nima:
            self.best_nima_epoch, self.best_nima = 0, 0.0
        if self.args.is_test_psnr_ssim:
            self.best_psnr_epoch, self.best_psnr = 0, 0.0
            self.best_ssim_epoch, self.best_ssim = 0, 0.0

        # set loss functions 
        self.criterionPercep = PerceptualLoss()
        self.criterionIdt = MultiscaleRecLoss(scale=3, rec_loss_type=self.args.idt_loss_type, multiscale=True)      # help='identity_loss: l1|l2|smoothl1 '
        self.criterionGAN = GANLoss(self.args.adv_loss_type)                        # tensor 옵션은 꺼도 됨(rahinge에서는)adversarial Loss: ls|original|hinge|rahinge|rals
        
        # start from scratch or trained models
        if self.args.pretrained_model:
            start_step = int(self.args.pretrained_model * self.train_steps_per_epoch)
            self.load_pretrained_model(self.args.pretrained_model)
        else:
            start_step = 0
        
        # start training
        print("======================================= start training =======================================")
        self.start_time = time.time()
        total_steps = int(self.args.total_epochs * self.train_steps_per_epoch)
        self.val_start_steps = int(self.args.num_epochs_start_val * self.train_steps_per_epoch)
        self.val_each_steps = int(self.args.val_each_epochs * self.train_steps_per_epoch)
              
        print("=========== start to iteratively train generator and discriminator ===========")
        pbar = tqdm(total=total_steps, desc='Train epoches', initial=start_step)
        for step in range(start_step, total_steps):
            ########## model train
            self.G.train()
            self.D.train()

            ########## data iter
            input = next(self.fetcher)
            self.real_raw, self.real_exp, self.real_raw_name = input.img_raw, input.img_exp, input.img_name         # fetcher returns dictionary

            ########## forward
            self.fake_exp = self.G(self.real_raw)
            self.fake_exp_store = self.fake_exp_pool.query(self.fake_exp)

            ########## duhyeonkim updated : color sensitity applied(blue changes too further in og)
            # Human eye sensitivity weights for RGB channels
            # human_color_sensitivity = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).to(self.real_raw.device)
            # # Apply the weights to fake_exp_store after querying
            # self.real_raw_weighted          = self.real_raw * human_color_sensitivity  # Weighted RGB
            # self.fake_exp_weighted          = self.fake_exp * human_color_sensitivity
            # self.real_exp_weighted          = self.real_exp * human_color_sensitivity
            # self.fake_exp_store_weighted    = self.fake_exp_store * human_color_sensitivity

            ########## update D
            self.d_optimizer.zero_grad()
            real_exp_preds = self.D(self.real_exp)
            fake_exp_preds = self.D(self.fake_exp_store.detach())
            d_loss = self.criterionGAN(real_exp_preds, fake_exp_preds, None, None, for_discriminator=True)      # rahinge loss여서 None, None
            if self.args.adv_input:                 # default=True
                input_preds = self.D(self.real_raw)
                d_loss += self.criterionGAN(real_exp_preds, input_preds, None, None, for_discriminator=True)
            d_loss.backward()
            self.d_optimizer.step()
            self.d_loss = d_loss.item()             # for logging and printing

            ########## update G
            self.g_optimizer.zero_grad()
            real_exp_preds = self.D(self.real_exp)
            fake_exp_preds = self.D(self.fake_exp)

            # -----------     GAN loss      -----------
            g_adv_loss = self.args.lambda_adv * self.criterionGAN(real_exp_preds, fake_exp_preds, None, None, for_discriminator=False)
            self.g_adv_loss = g_adv_loss.item()
            g_loss = g_adv_loss

            # -----------   Fidelity loss   -----------
            g_percep_loss = self.args.lambda_percep * self.criterionPercep((self.fake_exp+1.)/2., (self.real_raw+1.)/2.)
            self.g_percep_loss = g_percep_loss.item()
            g_loss += g_percep_loss
            
            # -----------   Identity loss   -----------
            self.real_exp_idt = self.G(self.real_exp)
            g_idt_loss = self.args.lambda_idt * self.criterionIdt(self.real_exp_idt, self.real_exp)             # duhyeonkim updated : 여기는 weighted 안했음
            self.g_idt_loss = g_idt_loss.item()
            g_loss += g_idt_loss
            
            g_loss.backward()
            self.g_optimizer.step()
            self.g_loss = g_loss.item()

            ### print info and save models
            self.print_info(step, total_steps, pbar)

            ### logging using tensorboard
            self.logging(step)

            ### validation 
            self.model_validation(step)                 # for unpaired data -> impossible to calculate PSNR and SSIM
            
            ### learning rate update
            if step % self.train_steps_per_epoch == 0:
                current_epoch = step // self.train_steps_per_epoch
                self.lr_scheduler_g.step()              # duhyeonkim updated : optimizer epoch option Deprecated only .step() used
                self.lr_scheduler_d.step()              # duhyeonkim updated : optimizer epoch option Deprecated only .step() used
                for param_group in self.g_optimizer.param_groups:
                    pbar.write("====== Epoch: {:>3d}/{}, Learning rate(lr) of Encoder(E) and Generator(G): [{}], ".format(((step + 1) // self.train_steps_per_epoch), self.args.total_epochs, param_group['lr']), end='')
                for param_group in self.d_optimizer.param_groups:
                    pbar.write("Learning rate (lr) of Discriminator(D): [{}] ======".format(param_group['lr']))
                
                # duhyeonkim updated : i have unbalanced dataset (so i wanted to reload and shuffle the unseen data)
                self.loaders.ref =get_train_loader(  root=self.args.train_img_dir,                    # default='./data/fivek/train'
                                                    img_size=self.args.image_size,                   # default=512
                                                    resize_size=self.args.resize_size,               # default=256
                                                    batch_size=self.args.train_batch_size,           # default=10
                                                    shuffle=self.args.shuffle,                       # default=False
                                                    num_workers=self.args.num_workers,               # 8 https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
                                                    drop_last=self.args.drop_last)

                self.fetcher = InputFetcher(self.loaders.ref)

            pbar.update(1)
            pbar.set_description(f"Train epoch %.2f" % ((step+1.0)/self.train_steps_per_epoch))
        
        self.val_best_results()

        pbar.write("=========== Complete training ===========")
        pbar.close()

    
    # duhyeonkim updated : image loss dictionary memory optimized
    def logging(self, step):
        self.loss.clear()
        self.images.clear()

        self.loss.update({
            'D/Total': self.d_loss,
            'G/Total': self.g_loss,
            'G/adv_loss': self.g_adv_loss,
            'G/percep_loss': self.g_percep_loss,
            'G/idt_loss': self.g_idt_loss
        })

        real_exp = denorm(self.real_exp.detach().cpu())
        real_exp_idt = denorm(self.real_exp_idt.detach().cpu())
        real_raw = denorm(self.real_raw.detach().cpu())
        fake_exp = denorm(self.fake_exp.detach().cpu())
        fake_exp_store = denorm(self.fake_exp_store.detach().cpu())

        self.images.update({
            'Train_realExpIdt/realExp_realExpIdt': torch.cat([real_exp, real_exp_idt], dim=3),
            'Train_compare/realRaw_fakeExp_realExp': torch.cat([real_raw, fake_exp, real_exp], dim=3),
            'Train_fakeExp/fakeExp': fake_exp,
            'Train_fakeExpStore/fakeExpStore': fake_exp_store
        })

        if (step+1) % self.args.log_step == 0:                          # log_step = 100(default) 
            if self.args.use_tensorboard:
                for tag, value in self.loss.items():
                    self.logger.scalar_summary(tag, value, step+1)
                for tag, image in self.images.items():
                    if image.dim() == 4:
                        self.logger.images_summary(tag, image, step + 1)
                    else:
                        self.logger.images_summary(tag, image, step + 1)

    
    def print_info(self, step, total_steps, pbar):
        current_epoch = (step+1) / self.train_steps_per_epoch

        if (step + 1) % self.args.info_step == 0:                           # info_step = 100(default)
            elapsed_num = time.time() - self.start_time
            elapsed = str(datetime.timedelta(seconds=elapsed_num))

            # duhyeonkim updataed : Improved readability
            pbar.write((
                            "Elapse:{:>.12s}, D_Step:{:>6d}/{}, G_Step:{:>6d}/{}, "
                            "D_loss:{:>.4f}, G_loss:{:>.4f}, G_percep_loss:{:>.4f}, "
                            "G_adv_loss:{:>.4f}, G_idt_loss:{:>.4f}"
                        ).format(
                            elapsed, step + 1, total_steps, (step + 1), total_steps,
                            self.d_loss, self.g_loss, self.g_percep_loss, 
                            self.g_adv_loss, self.g_idt_loss
                        ))

                    
        # sample images 
        if (step + 1) % self.args.sample_step == 0:                         # sample_step = 100(default)
            for i in range(0, self.real_raw.size(0)):
                save_imgs = torch.cat([
                    denorm(self.real_raw.detach())[i:i + 1, :, :, :],       # denorm returns (0,1) clamped value
                    denorm(self.fake_exp.detach())[i:i + 1, :, :, :], 
                    denorm(self.real_exp.detach())[i:i + 1, :, :, :]
                ], dim=3)
                save_image(
                    save_imgs, 
                    os.path.join(self.sample_path, 
                    '{:0>3.2f}_{:s}_{:0>2d}_realRaw_fakeExp_realExp.png'.format(
                        current_epoch, self.real_raw_name[i], i
                    ))
                )

        # duhyeonkim updated : compact and Pathlib used
        # save models
        if (step + 1) % self.model_save_step == 0:                          # every 1 epoch
            if self.args.parallel and torch.cuda.device_count() > 1:
                checkpoint = {
                    "G_net": self.G.module.state_dict(),
                    "D_net": self.D.module.state_dict(),
                    "epoch": current_epoch,
                    "g_optimizer": self.g_optimizer.state_dict(),
                    "d_optimizer": self.d_optimizer.state_dict(),
                    "lr_scheduler_g": self.lr_scheduler_g.state_dict(),
                    "lr_scheduler_d": self.lr_scheduler_d.state_dict()
                }
            else:
                checkpoint = {
                    "G_net": self.G.state_dict(),
                    "D_net": self.D.state_dict(),
                    "epoch": current_epoch,
                    "g_optimizer": self.g_optimizer.state_dict(),
                    "d_optimizer": self.d_optimizer.state_dict(),
                    "lr_scheduler_g": self.lr_scheduler_g.state_dict(),
                    "lr_scheduler_d": self.lr_scheduler_d.state_dict()
                }

            save_path = Path(self.model_save_path)
            save_path.mkdir(parents=True, exist_ok=True)

            model_filename = f"{self.args.version}_{self.args.adv_loss_type}_{current_epoch}.pth"
            model_filepath = save_path / model_filename

            try:
                torch.save(checkpoint, model_filepath)
                pbar.write(f"✅ Model checkpoint saved: {model_filepath}")
            except Exception as e:
                pbar.write(f"❌ Error saving model checkpoint: {e}")             


    def model_validation(self, step):
        if (step + 1) > self.val_start_steps:                           # start 8(epoch) every 2(epoch)
            if (step + 1) % self.val_each_steps == 0:
                val = {}
                current_epoch = (step + 1) / self.train_steps_per_epoch
                val_save_path = self.val_result_path + '/' + 'validation_' + str(current_epoch)
                val_compare_save_path = self.val_result_path + '/' + 'validation_compare_' + str(current_epoch)
                val_start = 0
                val_total_steps = len(self.loaders.val)
                
                if not os.path.exists(val_save_path):
                    os.makedirs(val_save_path)
                if not os.path.exists(val_compare_save_path):
                    os.makedirs(val_compare_save_path)

                self.G.eval()

                pbar = tqdm(total=(val_total_steps - val_start), desc='Validation epoches', position=val_start)
                pbar.write("============================== Start validation ==============================")
                with torch.no_grad():
                    for val_step in range(val_start, val_total_steps):
                        
                        input = next(self.fetcher_val)
                        val_real_raw, val_name = input.img_raw, input.img_name

                        val_fake_exp = self.G(val_real_raw)

                        # duhyeonkim updated : .data to .detach()
                        for i in range(0, denorm(val_real_raw.detach()).size(0)):
                            save_imgs = denorm(val_fake_exp.detach())[i:i + 1,:,:,:]
                            save_image(save_imgs, os.path.join(val_save_path, '{:s}_{:0>3.2f}_valFakeExp.png'.format(val_name[i], current_epoch)))

                            save_imgs_compare = torch.cat([denorm(val_real_raw.detach())[i:i + 1,:,:,:], denorm(val_fake_exp.detach())[i:i + 1,:,:,:]], 3)
                            save_image(save_imgs_compare, os.path.join(val_compare_save_path, '{:s}_{:0>3.2f}_valRealRaw_valFakeExp.png'.format(val_name[i], current_epoch)))

                        elapsed = time.time() - self.start_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        if val_step % self.args.info_step == 0:
                            pbar.write("=== Elapse:{}, Save {:>3d}-th val_fake_exp images into {} ===".format(elapsed, val_step, val_save_path))

                        # for tensorboard
                        val['val/valFakeExp'] = denorm(val_fake_exp.detach().cpu())
                        val['val_compare/valRealRaw_valFakeExp'] = torch.cat([denorm(val_real_raw.cpu()), denorm(val_fake_exp.detach().cpu())], 3)

                        pbar.update(1)

                        if self.args.use_tensorboard:
                            for tag, images in val.items():
                                self.logger.images_summary(tag, images, val_step + 1)

                    pbar.close()
                    if self.args.is_test_nima:
                        self.nima_result_save_path = './results/nima_val_results/'
                        curr_nima = calc_nima(val_save_path, self.nima_result_save_path,  current_epoch)
                        if self.best_nima < curr_nima:
                            self.best_nima = curr_nima
                            self.best_nima_epoch = current_epoch
                        print("====== Avg. NIMA: {:>.4f} ======".format(curr_nima))
                    
                    if self.args.is_test_psnr_ssim:
                        self.psnr_save_path = './results/psnr_val_results/' 
                        curr_psnr = calc_psnr(val_save_path, self.args.val_label_dir, self.psnr_save_path, current_epoch)
                        if self.best_psnr < curr_psnr:
                            self.best_psnr = curr_psnr
                            self.best_psnr_epoch = current_epoch
                        print("====== Avg. PSNR: {:>.4f} dB ======".format(curr_psnr))

                        self.ssim_save_path = './results/ssim_val_results/' 
                        curr_ssim = calc_ssim(val_save_path, self.args.val_label_dir, self.ssim_save_path, current_epoch)
                        if self.best_ssim < curr_ssim:
                            self.best_ssim = curr_ssim
                            self.best_ssim_epoch = current_epoch
                        print("====== Avg. SSIM: {:>.4f}  ======".format(curr_ssim))
                # torch.cuda.empty_cache()
                torch.mps.empty_cache()
                time.sleep(2)


    def val_best_results(self):
        if self.args.is_test_psnr_ssim:
            if not os.path.exists(self.psnr_save_path):
                os.makedirs(self.psnr_save_path)
            psnr_result = self.psnr_save_path + 'PSNR_total_results_epoch_avgpsnr.csv'
            psnrfile = open(psnr_result, 'a+')
            psnrfile.write('Best epoch: ' + str(self.best_psnr_epoch) + ',' + str(round(self.best_psnr, 6)) + '\n')
            psnrfile.close()

            if not os.path.exists(self.ssim_save_path):
                os.makedirs(self.ssim_save_path)
            ssim_result = self.ssim_save_path + 'SSIM_total_results_epoch_avgssim.csv'
            ssimfile = open(ssim_result, 'a+')
            ssimfile.write('Best epoch: ' + str(self.best_ssim_epoch) + ',' + str(round(self.best_ssim, 6)) + '\n')
            ssimfile.close()

        if self.args.is_test_nima:
            nima_total_result = self.nima_result_save_path + 'NIMA_total_results_epoch_mean_std.csv'
            totalfile = open(nima_total_result, 'a+')
            totalfile.write('Best epoch:' + str(self.best_nima_epoch) + ',' + str(round(self.best_nima, 6)) + '\n')
            totalfile.close()

    
    """define some functions"""
    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.args.g_conv_dim, self.args.g_norm_fun, self.args.g_act_fun, self.args.g_use_sn).to(self.device)
        self.D = Discriminator(self.args.d_conv_dim, self.args.d_norm_fun, self.args.d_act_fun, self.args.d_use_sn, self.args.adv_loss_type).to(self.device)
        if self.args.parallel:                      # default=False
            self.G.to(self.args.gpu_ids[0])
            self.D.to(self.args.gpu_ids[0])
            self.G = nn.DataParallel(self.G, self.args.gpu_ids)
            self.D = nn.DataParallel(self.D, self.args.gpu_ids)
        print("=== Models have been created ===")
        
        # print network
        if self.args.is_print_network:              # default=True (print model / parmas_num / params_size(MB))
            self.print_network(self.G, 'Generator')
            self.print_network(self.D, 'Discriminator')

        # init network
        if self.args.init_type:                     # default='orthogonal' (options 'normal|xavier|kaiming|orthogonal')
            self.init_weights(self.G, init_type=self.args.init_type, gain=0.02)
            self.init_weights(self.D, init_type=self.args.init_type, gain=0.02)

        # optimizer
        if self.args.optimizer_type == 'adam':
            # Adam optimizer
            self.g_optimizer = torch.optim.Adam(params=self.G.parameters(), lr=self.args.g_lr, betas=[self.args.beta1, self.args.beta2], weight_decay=0.0001)
            self.d_optimizer = torch.optim.Adam(params=self.D.parameters(), lr=self.args.d_lr, betas=[self.args.beta1, self.args.beta2], weight_decay=0.0001)
        elif self.args.optimizer_type == 'rmsprop':
            # RMSprop optimizer
            self.g_optimizer = torch.optim.RMSprop(params=self.G.parameters(), lr=self.args.g_lr, alpha=self.args.alpha)
            self.d_optimizer = torch.optim.RMSprop(params=self.D.parameters(), lr=self.args.d_lr, alpha=self.args.alpha)
        else:
            raise NotImplementedError("=== Optimizer [{}] is not found ===".format(self.args.optimizer_type))

        # learning rate decay
        if self.args.lr_decay:
            def lambda_rule(epoch):
                return 1.0 - max(0, epoch + 1 - self.args.lr_num_epochs_decay) / self.args.lr_decay_ratio           # epoch이랑 lr_num_epochs_decay를 조절을 잘 해야됨 (음수 schedule이 될 수도 있음)
            self.lr_scheduler_g = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=lambda_rule)
            self.lr_scheduler_d = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, lr_lambda=lambda_rule)
            print("=== Set learning rate decay policy for Generator(G) and Discriminator(D) ===")
        
        self.fake_exp_pool = ImagePool(self.args.pool_size)                     # image buffer (default = 50) : only initialize
    
    # duhyeonkim updated :  torch.nn.init should be worked in no grad + m.weight.data removed(not recommended command)
    @torch.no_grad()                
    def init_weights(self, net, init_type='kaiming', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight, gain=gain)
                elif init_type == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                elif init_type == 'kaiming_uniform':
                    torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in')
                elif init_type == 'orthogonal':                 # duhyeonkim updaged : "mps" device does not support torch.linalg.qr
                    if torch.backends.mps.is_available():
                        weight = m.weight.to("cpu")
                        torch.nn.init.orthogonal_(weight, gain=gain)
                        m.weight.data.copy_(weight.to(self.device))
                    else:
                        torch.nn.init.orthogonal_(m.weight, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('Initialization method [{}] is not implemented'.format(init_type))
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    torch.nn.init.normal_(m.weight, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)
            elif classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    torch.nn.init.normal_(m.weight, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)
        print("=== Initialize network with [{}] ===".format(init_type))
        net.apply(init_func)                        # recursively apply init_func for submodules (that is weight key)


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        # print(model)
        print("=== The number of parameters of the above model [{}] is [{}] or [{:>.4f}M] ===".format(name, num_params, num_params / 1e6))


    def load_pretrained_model(self, resume_epochs):
        checkpoint_path = os.path.join(self.model_save_path, '{}_{}_{}.pth'.format(self.args.version, self.args.adv_loss_type, resume_epochs))
        if torch.accelerator.is_available():
            # save on GPU, load on GPU
            checkpoint = torch.load(checkpoint_path)
            self.G.load_state_dict(checkpoint['G_net'])
            self.D.load_state_dict(checkpoint['D_net'])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
            self.lr_scheduler_g.load_state_dict(checkpoint['lr_scheduler_g'])
            self.lr_scheduler_d.load_state_dict(checkpoint['lr_scheduler_d'])
        else:
            # save on GPU, load on CPU
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            self.G.load_state_dict(checkpoint['G_net'])
            self.D.load_state_dict(checkpoint['D_net'])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
            self.lr_scheduler_g.load_state_dict(checkpoint['lr_scheduler_g'])
            self.lr_scheduler_d.load_state_dict(checkpoint['lr_scheduler_d'])

        print("=========== loaded trained models (epochs: {})! ===========".format(resume_epochs))


    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.logger = Logger(self.log_path)
    
    # duhyeonkim updatad : identity loss external defined in losses.py

    # def identity_loss(self, idt_loss_type):
    #     if idt_loss_type == 'l1':
    #         criterion = nn.L1Loss()
    #         return criterion
    #     elif idt_loss_type == 'smoothl1':
    #         criterion = nn.SmoothL1Loss()
    #         return criterion
    #     elif idt_loss_type == 'l2':
    #         criterion = nn.MSELoss()
    #         return criterion
    #     else:
    #         raise NotImplementedError("=== Identity loss type [{}] is not implemented. ===".format(self.args.idt_loss_type))
  