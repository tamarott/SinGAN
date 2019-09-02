from __future__ import print_function
import SinGAN.functions
import SinGAN.models
import argparse
import os
import random
from SinGAN.imresize import imresize
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from skimage import io as img
import numpy as np
from skimage import color
import math
import imageio
import matplotlib.pyplot as plt
from train import *
from config import get_arguments

def generate_gif(Gs,Zs,reals,NoiseAmp,opt,fps=10):

    in_s = torch.full(Zs[0].shape, 0, device=opt.device)
    images_cur = []
    count = 0

    for G,Z_opt,noise_amp,real in zip(Gs,Zs,NoiseAmp,reals):
        pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
        nzx = Z_opt.shape[2]
        nzy = Z_opt.shape[3]
        #pad_noise = 0
        #m_noise = nn.ZeroPad2d(int(pad_noise))
        m_image = nn.ZeroPad2d(int(pad_image))
        images_prev = images_cur
        images_cur = []
        if count == 0:
            z_rand = functions.generate_noise([1,nzx,nzy])
            z_rand = z_rand.expand(1,3,Z_opt.shape[2],Z_opt.shape[3])
            z_prev1 = 0.95*Z_opt +0.05*z_rand
            z_prev2 = Z_opt
        else:
            z_prev1 = 0.95*Z_opt +0.05*functions.generate_noise([opt.nc_z,nzx,nzy])
            z_prev2 = Z_opt

        for i in range(0,100,1):
            if count == 0:
                z_rand = functions.generate_noise([1,nzx,nzy])
                z_rand = z_rand.expand(1,3,Z_opt.shape[2],Z_opt.shape[3])
                diff_curr = opt.beta_animation*(z_prev1-z_prev2)+(1-opt.beta_animation)*z_rand
            else:
                diff_curr = opt.beta_animation*(z_prev1-z_prev2)+(1-opt.beta_animation)*(functions.generate_noise([opt.nc_z,nzx,nzy]))

            z_curr = opt.alpha_animation*Z_opt+(1-opt.alpha_animation)*(z_prev1+diff_curr)
            z_prev2 = z_prev1
            z_prev1 = z_curr

            if images_prev == []:
                I_prev = in_s
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev, 1 / opt.scale_factor, opt)
                I_prev = I_prev[:, :, 0:real.shape[2], 0:real.shape[3]]
                #I_prev = functions.upsampling(I_prev,reals[count].shape[2],reals[count].shape[3])
                I_prev = m_image(I_prev)
            if count < opt.animation_start_scale:
                z_curr = Z_opt

            z_in = noise_amp*z_curr+I_prev
            I_curr = G(z_in.detach(),I_prev)

            if (count == len(Gs)-1):
                I_curr = functions.denorm(I_curr).detach()
                I_curr = I_curr[0,:,:,:].cpu().numpy()
                I_curr = I_curr.transpose(1, 2, 0)*255
                I_curr = I_curr.astype(np.uint8)

            images_cur.append(I_curr)
        count += 1
    dir2save = functions.generate_dir2save(opt)
    try:
        os.makedirs(dir2save)
    except OSError:
        pass
    imageio.mimsave('%s/alpha=%f_beta=%f.gif' % (dir2save,opt.alpha_animation,opt.beta_animation),images_cur,fps=fps)
    del images_cur

def SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt,in_s=None,scale_v=1,scale_h=1,n=0,gen_start_scale=0,num_samples=50):
    #if torch.is_tensor(in_s) == False:
    if in_s is None:
        in_s = torch.full(reals[0].shape, 0, device=opt.device)
    images_cur = []
    for G,Z_opt,noise_amp in zip(Gs,Zs,NoiseAmp):
        pad1 = ((opt.ker_size-1)*opt.num_layer)/2
        m = nn.ZeroPad2d(int(pad1))
        nzx = (Z_opt.shape[2]-pad1*2)*scale_v
        nzy = (Z_opt.shape[3]-pad1*2)*scale_h

        images_prev = images_cur
        images_cur = []

        for i in range(0,num_samples,1):
            if n == 0:
                z_curr = functions.generate_noise([1,nzx,nzy])
                z_curr = z_curr.expand(1,3,z_curr.shape[2],z_curr.shape[3])
                z_curr = m(z_curr)
            else:
                z_curr = functions.generate_noise([opt.nc_z,nzx,nzy])
                z_curr = m(z_curr)

            if images_prev == []:
                I_prev = m(in_s)
                #I_prev = m(I_prev)
                #I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                #I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev,1/opt.scale_factor, opt)
                I_prev = I_prev[:,:,0:round(scale_v*reals[n].shape[2]),0:round(scale_h*reals[n].shape[3])]
                I_prev = m(I_prev)
                #I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                #I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])

            if n < gen_start_scale:
                z_curr = Z_opt

            z_in = noise_amp*(z_curr)+I_prev
            I_curr = G(z_in.detach(),I_prev)

            if n == len(reals)-1:
                if opt.mode == 'train':
                    dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out, opt.input_name[:-4], gen_start_scale)
                else:
                    dir2save = functions.generate_dir2save(opt)
                try:
                    os.makedirs(dir2save)
                except OSError:
                    pass
                plt.imsave('%s/%d.png' % (dir2save, i), functions.convert_image_np(I_curr.detach()), vmin=0,vmax=1)
                #plt.imsave('%s/%d_%d.png' % (dir2save,i,n),functions.convert_image_np(I_curr.detach()), vmin=0, vmax=1)
                #plt.imsave('%s/in_s.png' % (dir2save), functions.convert_image_np(in_s), vmin=0,vmax=1)
            images_cur.append(I_curr)
        n+=1
    return I_curr.detach()

def SinGAN_SR(opt,Gs,Zs,reals,NoiseAmp):
    mode = opt.mode
    in_scale, iter_num = functions.calc_init_scale(opt)
    opt.scale_factor = 1 / in_scale
    opt.scale_factor_init = 1 / in_scale
    opt.mode = 'SR_train'
    #opt.alpha = 100
    opt.stop_scale = 0
    dir2trained_model = functions.generate_dir2save(opt)
    if (os.path.exists(dir2trained_model)):
        #print('Trained model does not exist, training SinGAN for SR')
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
        opt.mode = mode
    else:
        SR_train(opt,Gs,Zs,reals,NoiseAmp)
        opt.mode = mode
    print('%f' % pow(in_scale,iter_num))
    Zs_sr = []
    reals_sr = []
    NoiseAmp_sr = []
    Gs_sr = []
    real = reals[-1]#read_image(opt)
    for j in range(1,iter_num+1,1):
        real_ = imresize(real,pow(1/opt.scale_factor,j),opt)
        real_ = real_[:, :, 0:int(pow(1 / opt.scale_factor, j) * real.shape[2]),0:int(pow(1 / opt.scale_factor, j) * real.shape[3])]
        reals_sr.append(real_)
        Gs_sr.append(Gs[-1])
        NoiseAmp_sr.append(NoiseAmp[-1])
        z_opt = torch.full(real_.shape, 0, device=opt.device)
        m = nn.ZeroPad2d(5)
        z_opt = m(z_opt)
        Zs_sr.append(z_opt)
    out = SinGAN_generate(Gs_sr, Zs_sr,reals_sr,NoiseAmp_sr, opt,in_s=reals_sr[0], num_samples=1)
    dir2save = functions.generate_dir2save(opt)
    plt.imsave('%s.png' % (dir2save), functions.convert_image_np(out.detach()), vmin=0,vmax=1)
    return

def SinGAN_animation(opt,Gs, Zs, reals, NoiseAmp):
    opt.min_size=20
    opt.mode = 'animation_train'
    real = functions.read_image(opt)
    functions.adjust_scales2image(real, opt)
    dir2trained_model = functions.generate_dir2save(opt)
    if (os.path.exists(dir2trained_model)):
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
        opt.mode = 'animation'
    else:
        #real = functions.read_image(opt)
        #functions.adjust_scales2image(real, opt)
        train(opt, Gs, Zs, reals, NoiseAmp)
        opt.mode = 'animation'
    for start_scale in range(0,3,1):
        for b in range(80, 100, 5):
            opt.animation_start_scale = start_scale
            opt.beta_animation = b/100
            generate_gif(Gs,Zs,reals,NoiseAmp,opt,10)
    return
'''
def SR(Gs,Zs,reals,NoiseAmp,opt,in_s,SR_factor,n,count):

    #l = nn.ReplicationPad2d((0,1,0,1))
    #real_down = functions.upsampling(real,real.shape[2]*scale,real.shape[3]*scale)

    #in_s = torch.full(real_down.shape, 0, device=opt.device)

    #in_s = functions.upsampling(real_down,real.shape[2]*scale*opt.scale_factor,real.shape[3]*scale*opt.scale_factor)
    #in_s = functions.upsampling(in_s,real_down.shape[2],real_down.shape[3])
    #vutils.save_image(functions.denorm(in_s),
    #                    '%s/samples/in_s.png' %  (opt.out_),
    #                    normalize=False)


    images_cur = []
    #count = 0

    for G,Z_opt,noise_amp in zip(Gs,Zs,NoiseAmp):
        count = count+1
        pad1 = ((opt.ker_size-1)*opt.num_layer)/2
        m = nn.ZeroPad2d(int(pad1))

        #if (n == 0):
        #    nzx = (Z_opt.shape[2]-pad1*2)+pad1*2
        #    nzy = (Z_opt.shape[3]-pad1*2)*retarget_factor+pad1*2
        #else:
        nzx = (Z_opt.shape[2]-pad1*2)*pow(SR_factor,count)
        nzy = (Z_opt.shape[3]-pad1*2)*pow(SR_factor,count)

        images_prev = images_cur
        images_cur = []
        #pad1 = 0#((opt.ker_size-1)*opt.num_layer)/2
        #pad2 = pad1
        #m = nn.ReplicationPad2d((math.ceil(pad1),math.floor(pad1),math.ceil(pad2),math.floor(pad2)))



  
        if n == 0:
            z_curr = functions.generate_noise([1,nzx,nzy],1,opt.noise_type,1,opt.device)
            z_curr = z_curr.expand(1,3,z_curr.shape[2],z_curr.shape[3])
            z_curr = m(z_curr)
            #pad1 = ((opt.ker_size-1)*opt.num_layer)/2
            #pad2 = pad1
            #m = nn.ReplicationPad2d((pad1,pad1,pad2,pad2))
            #m = nn.ZeroPad2d(int(pad1))
        else:
            z_curr = functions.generate_noise([opt.nc_z,nzx,nzy],1,opt.noise_type,opt.noise_scale,opt.device)
            z_curr = m(z_curr)
            #pad1 = ((opt.ker_size-1)*opt.num_layer)/2
            #pad2 = pad1
            #m = nn.ReplicationPad2d((math.ceil(pad1),math.floor(pad1),math.ceil(pad2),math.floor(pad2)))
            #m = nn.ZeroPad2d(int(pad1))
        #z_curr = Z_opt#torch.full([1,3,nzx,nzy], 0, device=opt.device)

        if images_prev == []:
            I_prev = in_s
            #I_prev = functions.upsampling(I_prev,I_prev.shape[2]/opt.scale_factor,I_prev.shape[3]/opt.scale_factor)
            #I_prev = functions.upsampling(I_prev,reals[n].shape[2],retarget_factor*reals[n].shape[3])
            #I_prev = functions.upsampling(I_prev,1/opt.scale_factor)
            I_prev = m(I_prev)
            #I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])
            z_curr = functions.generate_noise([opt.nc_z,nzx,nzy],1,opt.noise_type,opt.noise_scale,opt.device)
            z_curr = m(z_curr)
        #I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])
        else:
            I_prev = images_prev[0]#[i]
            #I_prev = torch2uint8(I_prev)
            I_prev = imresize(I_prev, SR_factor, opt)
            #I_prev = functions.upsampling(I_prev,I_prev.shape[2]/opt.scale_factor,I_prev.shape[3]/opt.scale_factor)
            #I_prev = functions.upsampling(I_prev,1/opt.scale_factor)
            #I_prev = functions.upsampling(I_prev,reals[n].shape[2]*SR_factor,reals[n].shape[3]*SR_factor)
            I_prev = m(I_prev)
            #I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])


        z_curr = functions.generate_noise([1,I_prev.shape[2]-2*pad1,I_prev.shape[3]-2*pad1],1,opt.noise_type,1,opt.device)
        z_curr = m(z_curr)
        z_in = I_prev+noise_amp*z_curr
        #if images_prev == []:
        #    z_in = noise_amp*(Z_opt)+I_prev
        #else:
        #    z_in = noise_amp*(z_curr)+I_prev
        #z_in = noise_amp*torch.full([1,3,nzx+2*pad1,nzy+2*pad1], 0, device=opt.device)+I_prev

        #prev_sol = functions.denorm(I_prev)
        #vutils.save_image(prev_sol,
        #        '%s/samples/prev_frame_%d_%d.png' %  (opt.out_,count,i),
        #        normalize=False)
        #prev_z_in = functions.denorm(z_in)
        #vutils.save_image(prev_z_in,
        #        '%s/samples/z_in_%d_%d.png' %  (opt.out_,count,i),
        #        normalize=True)


        #del I_prev

        #I_curr = G(z_in.detach())
        I_curr = G(z_in.detach(),I_prev)
        #I_curr = l(I_curr)
        I_curr = I_curr[:,:,0:round(reals[-1].shape[2]*pow(SR_factor,count)),0:round(reals[-1].shape[3]*pow(SR_factor,count))]
        #curr_sol = functions.denorm(I_curr.detach())
        #curr_sol = functions.denorm2image(I_curr.detach(),functions.denorm(real))
        #curr_sol = functions.denorm(I_curr.detach())
        #curr_sol = (curr_sol-curr_sol.min())/(curr_sol.max()-curr_sol.min())
        #if (n == len(Gs)):

        #vutils.save_image(curr_sol,
        #    '%s/curr_frame_%d_%d.png' %  (opt.out_,n,i),
        #    normalize=True)

        plt.imsave('%s/curr_frame_%d_%d.png' %  (opt.out_2,n,i),
            functions.convert_image_np(I_curr.detach()), vmin=0, vmax=1)

        images_cur.append(I_curr)

        #del I_curr

        n+=1
    return I_curr.detach(),count


if __name__ == '__main__':
    opt = get_arguments()
    real_ = functions.read_image(opt)
    functions.adjust_scales2image(real_,opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    #start_scales = 0
    #external_AMP = 1

    Gs,Zs,reals,NoiseAmp = functions.load_trained_pyramid(opt)

    random_sample(Gs,Zs,reals,NoiseAmp,opt)
'''

### main generate giff ###
'''
fold = os.listdir('Input/balloons/')
opt = get_arguments()
opt.nfc_init = opt.nfc

for i in range(len(fold)):
    opt.ref_image = 'balloons/%s' % fold[i]
    image_name = '%s' % fold[i]
    #opt.ref_image = 'balloons/flock-of-seabirds.jpg'
    #image_name = 'flock-of-seabirds.jpg'
    
    real_ = read_image2np(opt)
    opt.num_scales = int((math.log(math.pow(opt.min_size/(real_.shape[0]),1),opt.scale_factor)))+1
    #opt.num_scales = 10#int((math.log(30/min([real_.shape[0],real_.shape[1]]),opt.scale_factor)))+1
    reals = []
    
    for i in range(0,opt.stop_scale+1,1):
        scale = pow(opt.scale_factor,opt.stop_scale-i)
        curr_real = imresize2.imresize2(real_,scale_factor=scale, kernel=None, antialiasing=True, kernel_shift_flag=False)
        curr_real = np2torch(curr_real)
        reals.append(curr_real)
    real = np2torch(real_)
    
    opt.out_ = '%s/%s/%f/%d' % (opt.out,image_name[:-4],opt.small_size,opt.num_scales)
    Gs = []
    Zs = []
    NoiseAmp = []
    
    #for i in range(0,1,opt.stop_scale):
    #    #opt.nfc = min(opt.nfc_init*pow(2,round(i/10)),256)
    #    #opt.min_nfc = min(opt.nfc_init*pow(2,round(i/10)),256)
    #    opt.outf = '%s/%d' % (opt.out_,i)
    #    netD,netG = init_models(opt)
    #    del netD
    #    netG.load_state_dict(torch.load('%s/netG.pth' % opt.outf))
    #    #netG.eval()
    #    Gs.append(netG)
    #    z_opt=torch.load('%s/z_opt.pth' % opt.outf)
    #    Zs.append(z_opt)
    #    noise_amp = opt.noise_amp/pow(1/opt.scale_factor,i)
    #    NoiseAmp.append(noise_amp)
    #    del z_opt
    #    del netG
    #    del noise_amp
    #torch.save(Zs, '%s/Zs.pth' % (opt.out_))
    #torch.save(Gs, '%s/Gs.pth' % (opt.out_))
    #torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))
    
    Gs = torch.load('%s/Gs.pth' % opt.out_)
    Zs = torch.load('%s/Zs.pth' % opt.out_)
    reals = torch.load('%s/reals.pth' % opt.out_)
    NoiseAmp = torch.load('%s/NoiseAmp.pth' % opt.out_)
    
    try:
        os.makedirs('%s/samples' % opt.out_ )
        os.makedirs('%s/gifs' % opt.out_ )
    except OSError:
        pass
    
    #random_sample(Gs,Zs,reals,NoiseAmp,opt)
    
    fixed_scales = 2
    opt.out_ = '%s/gifs_new/%d' % (opt.out_,fixed_scales)
    try:
        os.makedirs('%s' % opt.out_ )
    except OSError:
        pass
    
    for alpha in range(10,15,5):
        for beta in range(10,100,10):
            generate_gif(Gs,Zs,NoiseAmp,opt,alpha/100,beta/100,10,fixed_scales)

#del netD
#del netG

'''

### main generate samples ###
'''
fold = os.listdir('Input/colorful_buildings/')
opt = get_arguments()
opt.nfc_init = opt.nfc

for i in range(len(fold)):
    opt.ref_image = 'colorful_buildings/%s' % fold[i]
    image_name = '%s' % fold[i]
    #opt.ref_image = 'balloons/flock-of-seabirds.jpg'
    #image_name = 'flock-of-seabirds.jpg'

    real_ = read_image2np(opt)
    opt.num_scales = int((math.log(math.pow(opt.min_size/(real_.shape[0]),1),opt.scale_factor)))+1
    #opt.num_scales = 10#int((math.log(30/min([real_.shape[0],real_.shape[1]]),opt.scale_factor)))+1
    reals = []

    for i in range(0,opt.stop_scale+1,1):
        scale = pow(opt.scale_factor,opt.stop_scale-i)
        curr_real = imresize2.imresize2(real_,scale_factor=scale, kernel=None, antialiasing=True, kernel_shift_flag=False)
        curr_real = np2torch(curr_real)
        reals.append(curr_real)
    real = np2torch(real_)

    opt.out_ = '%s/%s/%f/%d' % (opt.out,image_name[:-4],opt.small_size,opt.num_scales)
    Gs = []
    Zs = []
    NoiseAmp = []

    #for i in range(0,1,opt.stop_scale):
    #    #opt.nfc = min(opt.nfc_init*pow(2,round(i/10)),256)
    #    #opt.min_nfc = min(opt.nfc_init*pow(2,round(i/10)),256)
    #    opt.outf = '%s/%d' % (opt.out_,i)
    #    netD,netG = init_models(opt)
    #    del netD
    #    netG.load_state_dict(torch.load('%s/netG.pth' % opt.outf))
    #    #netG.eval()
    #    Gs.append(netG)
    #    z_opt=torch.load('%s/z_opt.pth' % opt.outf)
    #    Zs.append(z_opt)
    #    noise_amp = opt.noise_amp/pow(1/opt.scale_factor,i)
    #    NoiseAmp.append(noise_amp)
    #    del z_opt
    #    del netG
    #    del noise_amp
    #torch.save(Zs, '%s/Zs.pth' % (opt.out_))
    #torch.save(Gs, '%s/Gs.pth' % (opt.out_))
    #torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

    Gs = torch.load('%s/Gs.pth' % opt.out_)
    Zs = torch.load('%s/Zs.pth' % opt.out_)
    reals = torch.load('%s/reals.pth' % opt.out_)
    NoiseAmp = torch.load('%s/NoiseAmp.pth' % opt.out_)

    #try:
    #    os.makedirs('%s/samples' % opt.out_ )
    #    os.makedirs('%s/gifs' % opt.out_ )
    #except OSError:
    #    pass

    #random_sample(Gs,Zs,reals,NoiseAmp,opt)

    fixed_scales = 0
    external_AMP = 1
    try:
        os.makedirs('%s/samples_Amp_%f_%d' % (opt.out_,external_AMP,fixed_scales) )
        #os.makedirs('%s/gifs' % opt.out_ )
    except OSError:
        pass

    random_sample(Gs,Zs,reals,NoiseAmp,fixed_scales,external_AMP,opt)


#del netD
#del netG

'''

### main SR###
'''
fold = os.listdir('Input/BSD100/')
opt = get_arguments()
#opt.ref_image = 'SR/33039_LR.png'
#image_name = '33039_LR.png'
opt.out_= opt.out
for i in range(len(fold)):
    opt.ref_image = 'BSD100/%s' % fold[i]
    image_name = '%s' % fold[i]

    Gs = []
    Zs = []
    NoiseAmp = []

    real_ = read_image2np(opt)

    #opt.num_scales = int((math.log(math.pow(opt.min_size/(min(real_.shape[0],real_.shape[1])),1),opt.scale_factor)))+1
    opt.num_scales = int((math.log(math.pow(opt.min_size/((real_.shape[0])),1),opt.scale_factor)))+1
    #opt.num_scales = int((math.log(math.pow(opt.min_size/((real_.shape[1])),1),opt.scale_factor)))+1
    scale2stop = int(math.log(min([250,max([real_.shape[0],real_.shape[1]])])/max([real_.shape[0],real_.shape[1]]),opt.scale_factor))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = imresize2.imresize2(real_,scale_factor=opt.scale1, kernel=None, antialiasing=True, kernel_shift_flag=False)
    #opt.scale_factor = math.pow(opt.min_size/(min(real_.shape[0],real_.shape[1])),1/(opt.stop_scale))
    #opt.scale_factor = math.pow(opt.min_size/((real_.shape[0])),1/(opt.stop_scale))
    #opt.scale_factor = math.pow(opt.min_size/((real_.shape[1])),1/(opt.stop_scale))
    #scale2stop = int(math.log(min([250,max([real_.shape[0],real_.shape[1]])])/max([real_.shape[0],real_.shape[1]]),opt.scale_factor))
    opt.stop_scale = opt.num_scales - scale2stop



    opt.out = '%s/%s/%f/%d' % (opt.out_,image_name[:-4],opt.small_size,opt.num_scales)

    Gs = torch.load('%s/Gs.pth' % opt.out)
    Zs = torch.load('%s/Zs.pth' % opt.out)
    NoiseAmp = torch.load('%s/NoiseAmp.pth' % opt.out)
    reals = torch.load('%s/reals.pth' % opt.out)

    n =opt.stop_scale
    real = reals[n]


    SR_factor = 4
    Num_scales = 2
    method = 'only'

    in_s = real
    ref = imresize2.imresize2(torch2uint8(in_s),scale_factor=SR_factor, kernel=None, antialiasing=True, kernel_shift_flag=False)
    ref = np2torch(ref)

    #ref = functions.upsampling(in_s,SR_factor*in_s.shape[2],SR_factor*in_s.shape[3])
    N = round(math.log(1/SR_factor,opt.scale_factor))
    r = pow((SR_factor),1/N)
    #N_ = int(N/Num_scales)

    if method == 'all':
        Gs = Gs[n-Num_scales+1:]
        Zs = Zs[n-Num_scales+1:]
        NoiseAmp = NoiseAmp[n-Num_scales+1:]
        Num_iter = N/Num_scales
    if method == 'only':
        Gs = Gs[n-Num_scales+1:n-Num_scales+2]
        Zs = Zs[n-Num_scales+1:n-Num_scales+2]
        NoiseAmp = NoiseAmp[n-Num_scales+1:n-Num_scales+2]
        Num_iter = N
    count = 0
    for i in range(0,int(Num_iter),1):

        opt.out_2 = '%s/SR_withGEN/%d_%d_%s/%d/%d' % (opt.out,SR_factor,Num_scales,method,n,i)

        try:
            os.makedirs('%s' % opt.out_2)
        except OSError:
            pass

        #l = nn.ReplicationPad2d((0,1,0,1))
        #in_s = l(in_s)
        #in_s = functions.upsampling(in_s,(r)*in_s.shape[2],(r)*in_s.shape[3])
        #in_s = in_s[:,:,0:round(real.shape[2]*pow(r,i+1)),0:round(real.shape[3]*pow(r,i+1))]
        in_s = imresize2.imresize2(torch2uint8(in_s),scale_factor=r, kernel=None, antialiasing=True, kernel_shift_flag=False)
        in_s = np2torch(in_s)

        plt.imsave('%s/in_s.png' %  (opt.out_2),
                    functions.convert_image_np(in_s), vmin=0, vmax=1)


        in_s,count = SR(Gs,Zs,reals,NoiseAmp,opt,in_s,r,n-(Num_scales-1),count)


    in_s = in_s[:,:,0:ref.shape[2],0:ref.shape[3]]
    plt.imsave('%s/Out.png' %  (opt.out_2),
                        functions.convert_image_np(in_s), vmin=0, vmax=1)
    plt.imsave('%s/ref.png' %  (opt.out_2),
                        functions.convert_image_np(ref), vmin=0, vmax=1)
    plt.imsave('Out/SR_BSD100_%d/%f/%d_%s/%s.png' % (opt.alpha,opt.scale_factor,Num_scales,method,image_name[:-7]),
                        functions.convert_image_np(in_s), vmin=0, vmax=1)

    #del netD
    #del netG

'''

### main paint2im###
'''
fold = os.listdir('Input/New/')

opt = get_arguments()
#for i in range(len(fold)):
#opt.ref_image = 'balloons/%s' % fold[i]
#image_name = '%s' % fold[i]
#opt.ref_image = 'balloons/%s' % fold[i]
#image_name = '%s' % fold[i]

opt.ref_image = 'New/trees3.jpg'
image_name = 'trees3.jpg'

real_ = read_image2np(opt)

opt.niter_init = opt.niter
opt.noise_amp_init = opt.noise_amp
opt.nfc_init = opt.nfc
opt.min_nfc_init = opt.min_nfc
opt.scale_factor_init = opt.scale_factor

#opt.num_scales = int((math.log(math.pow(opt.min_size/(min(real_.shape[0],real_.shape[1])),1),opt.scale_factor_init)))+1
opt.num_scales = int((math.log(math.pow(opt.min_size/(real_.shape[0]),1),opt.scale_factor_init)))+1
opt.out_ = '%s/%s/%f/%d' % (opt.out,image_name[:-4],opt.small_size,opt.num_scales)
scale2stop = int(math.log(min([250,max([real_.shape[0],real_.shape[1]])])/max([real_.shape[0],real_.shape[1]]),opt.scale_factor_init))
opt.stop_scale = opt.num_scales - scale2stop
opt.scale1 = min(250/max([real_.shape[0],real_.shape[1]]),1)
real = imresize2.imresize2(real_,scale_factor=opt.scale1, kernel=None, antialiasing=True, kernel_shift_flag=False)
opt.scale_factor = math.pow(opt.min_size/(real.shape[0]),1/(opt.stop_scale))
#opt.scale_factor = math.pow(opt.min_size/(min(real_.shape[0],real_.shape[1])),1/(opt.stop_scale))
scale2stop = int(math.log(min([250,max([real_.shape[0],real_.shape[1]])])/max([real_.shape[0],real_.shape[1]]),opt.scale_factor_init))
opt.stop_scale = opt.num_scales - scale2stop

opt.out_ = '%s/%s/%f/%d' % (opt.out,image_name[:-4],opt.small_size,opt.num_scales)
opt.out_new = opt.out_
Gs = []
Zs = []
NoiseAmp = []

#opt.nfc_init = opt.nfc
#opt.min_nfc_init = opt.min_nfc
#for i in range(opt.num_scales-1):
#    opt.nfc = min(opt.nfc_init*pow(2,math.floor(i /4)),128)
#    opt.min_nfc = min(opt.min_nfc_init*pow(2,math.floor(i /4)),128)
#    opt.outf = '%s/%d' % (opt.out_,i)
#    netD,netG = init_models(opt)
#    netG.load_state_dict(torch.load('%s/netG.pth' % opt.outf))
#    netG.eval()
#    Gs.append(netG)
#    z_opt=torch.load('%s/z_opt.pth' % opt.outf)
#    Zs.append(z_opt)
#    noise_amp = opt.noise_amp/pow(pow(1/opt.scale_factor,0.5),i)
#    NoiseAmp.append(noise_amp)

Gs = torch.load('%s/Gs.pth' % opt.out_)
Zs = torch.load('%s/Zs.pth' % opt.out_)
NoiseAmp = torch.load('%s/NoiseAmp.pth' % opt.out_)
reals = torch.load('%s/reals.pth' % opt.out_)
retarget_factor_x = 1
retarget_factor_y = 1


#n = 8
for n in range(1,10,1):
    Gs_ = Gs[n:]
    Zs_ = Zs[n:]
    NoiseAmp_ = NoiseAmp[n:]

    #real = read_image(opt)
    real = reals[n]
    #real = torch.cat((real[:,:,:,int(real.shape[3]/2+1):real.shape[3]],real[:,:,:,1:int(real.shape[3]/2)]),3)

    paint = img.imread('Input/paint/trees3_paint.png')
    paint = paint[:,:,0:3]
    paint = np2torch(paint)
    paint = imresize2.imresize2(torch2uint8(paint),output_shape=[reals[-1].shape[2]*retarget_factor_y,reals[-1].shape[3]*retarget_factor_x], kernel=None, antialiasing=True, kernel_shift_flag=False)
    paint = np2torch(paint)

    #paint = paint[:,:,:,None]
    #paint = paint.transpose((3, 2, 0, 1))/255
    #paint = torch.from_numpy(paint)
    #paint = functions.move_to_gpu(paint)
    #paint = paint.type(torch.cuda.FloatTensor)
    #paint = functions.norm(paint)

    #scale = pow(opt.scale_factor,opt.num_scales-1-n)
    #real_down = functions.upsampling(real,real.shape[2]*scale,retarget_factor*real.shape[3]*scale)

    scale = pow(opt.scale_factor,opt.stop_scale-n+1)
    #paint = paint + 0.2*functions.generate_noise([3,paint.shape[2],paint.shape[3]],1,opt.noise_type,4,opt.device)
    in_s = imresize2.imresize2(torch2uint8(paint),scale_factor=scale, kernel=None, antialiasing=True, kernel_shift_flag=False)
    #in_s = imresize2.imresize2(in_s,output_shape=[in_s.shape[0]*retarget_factor_y,in_s.shape[1]*retarget_factor_x], kernel=None, antialiasing=True, kernel_shift_flag=False)
    in_s = imresize2.imresize2(in_s,scale_factor=1/opt.scale_factor, kernel=None, antialiasing=True, kernel_shift_flag=False)
    in_s = imresize2.imresize2(in_s,output_shape=[reals[n].shape[2]*retarget_factor_y,reals[n].shape[3]*retarget_factor_x], kernel=None, antialiasing=True, kernel_shift_flag=False)
    in_s = np2torch(in_s)
    in_s = in_s[:,:,0:int(reals[n].shape[2]*retarget_factor_y),0:int(reals[n].shape[3]*retarget_factor_x)]
    real_s = np2torch(imresize2.imresize2(torch2uint8(reals[n]),output_shape=[reals[n].shape[2]*retarget_factor_y,reals[n].shape[3]*retarget_factor_x], kernel=None, antialiasing=True, kernel_shift_flag=False))
    criterion = nn.MSELoss()
    amp_in = torch.sqrt(criterion(in_s, real_s))
    in_s = in_s + 0.1*amp_in*functions.generate_noise([1,in_s.shape[2],in_s.shape[3]],1,opt.noise_type,1,opt.device)
    target = functions.convert_image_np(in_s)
    #target = target[:,:,1]
    source = functions.convert_image_np(reals[n])
    #source = source[:,:,1]
    temp = np.zeros(target.shape)
    for i in range(0,3,1):
        temp[:,:,i] = functions.hist_match(target[:,:,i],source[:,:,i])
    in_s = np2torch(255*temp)

    #in_s = functions.upsampling(paint,retarget_factor_y*reals[n-1].shape[2],retarget_factor_x*reals[n-1].shape[3])
    #in_s = functions.upsampling(in_s,retarget_factor_y*reals[n].shape[2],retarget_factor_x*reals[n].shape[3])



    #in_s = functions.upsampling(in_s,real_down.shape[2],real_down.shape[3])
    #in_s = functions.upsampling(real_down,real_down.shape[2]*opt.scale_factor,real_down.shape[3]*opt.scale_factor)
    #in_s = functions.upsampling(in_s,real_down.shape[2],real_down.shape[3])
    #in_s = torch.full(real_down.shape, 0, device=opt.device)


    ### edit ###
    #in_s = real_down.clone()
    #length = int(real_down.shape[3]/2)-1
    #in_s[:,:,:,0:length] = real_down[:,:,:,int(real_down.shape[3]/2):int(real_down.shape[3]/2)+length]
    #in_s[:,:,:,length:2*length] = real_down[:,:,:,0:length]
    #in_s = functions.upsampling(in_s,real_down.shape[2]*opt.scale_factor,real_down.shape[3]*opt.scale_factor)

    #if n == 0:
    #    in_s = torch.full([real_down.shape[0],real_down.shape[1],real_down.shape[2]*opt.scale_factor,real_down.shape[3]*opt.scale_factor], 0, device=opt.device)
    #else:
    #    in_s = functions.upsampling(real_down,real_down.shape[2]*opt.scale_factor,real_down.shape[3]*opt.scale_factor)


    opt.out_ = '%s/paint2im_noise2/trees3_paint/%d' % (opt.out_new,n)
    try:
        os.makedirs('%s' % opt.out_ )
    except OSError:
        pass

    #vutils.save_image(functions.denorm(in_s),
    #                    '%s/in_s.png' %  (opt.out_),
    #                    normalize=False)
    plt.imsave('%s/in_s.png' %  (opt.out_),
                    functions.convert_image_np(in_s), vmin=0, vmax=1)
    plt.imsave('%s/in.png' %  (opt.out_),
                    functions.convert_image_np(paint), vmin=0, vmax=1)

    #vutils.save_image(functions.denorm(real_down),
    #                    '%s/in_down.png' %  (opt.out_),
    #                    normalize=False)
    #vutils.save_image(functions.denorm(real),
    #                    '%s/in.png' %  (opt.out_),
    #                    normalize=False)

    retarget2(Gs_,Zs_,reals,NoiseAmp_,opt,in_s,retarget_factor_x,retarget_factor_y,n)
#del netD
#del netG

'''

### main harmonization###
'''

n_im ='23_2'

fold = os.listdir('Input/harmonization_target_2/')

opt = get_arguments()
#for i in range(len(fold)):
#opt.ref_image = 'balloons/%s' % fold[i]
#image_name = '%s' % fold[i]
#opt.ref_image = 'balloons/%s' % fold[i]
#image_name = '%s' % fold[i]

opt.ref_image = 'harmonization_target_2/23_target.jpg' #%(n_im)
image_name = '23_target.jpg' #%(n_im)


real_ = read_image2np(opt)

opt.niter_init = opt.niter
opt.noise_amp_init = opt.noise_amp
opt.nfc_init = opt.nfc
opt.min_nfc_init = opt.min_nfc
opt.scale_factor_init = opt.scale_factor

#opt.num_scales = int((math.log(math.pow(opt.min_size/(min(real_.shape[0],real_.shape[1])),1),opt.scale_factor_init)))+1
opt.num_scales = int((math.log(math.pow(opt.min_size/(real_.shape[0]),1),opt.scale_factor_init)))+1
opt.out_ = '%s/%s/%f/%d' % (opt.out,image_name[:-4],opt.small_size,opt.num_scales)
scale2stop = int(math.log(min([250,max([real_.shape[0],real_.shape[1]])])/max([real_.shape[0],real_.shape[1]]),opt.scale_factor_init))
opt.stop_scale = opt.num_scales - scale2stop
opt.scale1 = min(250/max([real_.shape[0],real_.shape[1]]),1)
real = imresize2.imresize2(real_,scale_factor=opt.scale1, kernel=None, antialiasing=True, kernel_shift_flag=False)
opt.scale_factor = math.pow(opt.min_size/(real.shape[0]),1/(opt.stop_scale))
#opt.scale_factor = math.pow(opt.min_size/(min(real_.shape[0],real_.shape[1])),1/(opt.stop_scale))
scale2stop = int(math.log(min([250,max([real_.shape[0],real_.shape[1]])])/max([real_.shape[0],real_.shape[1]]),opt.scale_factor_init))
opt.stop_scale = opt.num_scales - scale2stop



opt.out_ = '%s/%s/%f/%d' % (opt.out,image_name[:-4],opt.small_size,opt.num_scales)
opt.out_new = opt.out_
Gs = []
Zs = []
NoiseAmp = []

#opt.nfc_init = opt.nfc
#opt.min_nfc_init = opt.min_nfc
#for i in range(opt.num_scales-1):
#    opt.nfc = min(opt.nfc_init*pow(2,math.floor(i /4)),128)
#    opt.min_nfc = min(opt.min_nfc_init*pow(2,math.floor(i /4)),128)
#    opt.outf = '%s/%d' % (opt.out_,i)
#    netD,netG = init_models(opt)
#    netG.load_state_dict(torch.load('%s/netG.pth' % opt.outf))
#    netG.eval()
#    Gs.append(netG)
#    z_opt=torch.load('%s/z_opt.pth' % opt.outf)
#    Zs.append(z_opt)
#    noise_amp = opt.noise_amp/pow(pow(1/opt.scale_factor,0.5),i)
#    NoiseAmp.append(noise_amp)

Gs = torch.load('%s/Gs.pth' % opt.out_)
Zs = torch.load('%s/Zs.pth' % opt.out_)
NoiseAmp = torch.load('%s/NoiseAmp.pth' % opt.out_)
reals = torch.load('%s/reals.pth' % opt.out_)

for n in range(1,9,1):
    Gs_ = Gs[n:]
    Zs_ = Zs[n:]
    NoiseAmp_ = NoiseAmp[n:]

    #real = read_image(opt)
    real = reals[n]
    #real = torch.cat((real[:,:,:,int(real.shape[3]/2+1):real.shape[3]],real[:,:,:,1:int(real.shape[3]/2)]),3)

    opt2 = opt
    opt2.ref_image = 'harmonization_target_2/%s_naive.jpg'  %(n_im)
    paint = read_image2np(opt2)
    paint = paint.astype(np.uint8)
    paint = imresize2.imresize2(paint,output_shape=[reals[-1].shape[2],reals[-1].shape[3]])

    paint = paint[:,:,0:3]
    paint = np2torch(paint)

    opt2.ref_image = 'harmonization_target_2/%s_c_mask_dilated.jpg'  %(n_im)
    mask = read_image2np(opt2)
    mask = mask[:,:,0:3]
    mask = mask.astype(np.uint8)
    mask = imresize2.imresize2(mask,output_shape=[reals[-1].shape[2],reals[-1].shape[3]])
    mask = np.round_(mask,0)
    mask = np2torch(mask)
    mask = functions.denorm(mask)


    #paint = paint[:,:,:,None]
    #paint = paint.transpose((3, 2, 0, 1))/255
    #paint = torch.from_numpy(paint)
    #paint = functions.move_to_gpu(paint)
    #paint = paint.type(torch.cuda.FloatTensor)
    #paint = functions.norm(paint)

    #opt.num_scales = int((math.log(opt.min_sizemin([real.shape[0],real.shape[1]]),opt.scale_factor)))+1
    #scale2stop = int(math.log(min([250,max([real.shape[0],real.shape[1]])])/max([real.shape[0],real.shape[1]]),opt.scale_factor))
    #opt.stop_scale = opt.num_scales - scale2stop

    retarget_factor_x = 1
    retarget_factor_y = 1
    #scale = pow(opt.scale_factor,opt.num_scales-1-n)
    #real_down = functions.upsampling(real,real.shape[2]*scale,retarget_factor*real.shape[3]*scale)

    in_s = functions.upsampling(paint,retarget_factor_y*reals[n-1].shape[2],retarget_factor_x*reals[n-1].shape[3])
    in_s = functions.upsampling(in_s,retarget_factor_y*reals[n].shape[2],retarget_factor_x*reals[n].shape[3])


    #in_s = functions.upsampling(in_s,real_down.shape[2],real_down.shape[3])
    #in_s = functions.upsampling(real_down,real_down.shape[2]*opt.scale_factor,real_down.shape[3]*opt.scale_factor)
    #in_s = functions.upsampling(in_s,real_down.shape[2],real_down.shape[3])
    #in_s = torch.full(real_down.shape, 0, device=opt.device)


    ### edit ###
    #in_s = real_down.clone()
    #length = int(real_down.shape[3]/2)-1
    #in_s[:,:,:,0:length] = real_down[:,:,:,int(real_down.shape[3]/2):int(real_down.shape[3]/2)+length]
    #in_s[:,:,:,length:2*length] = real_down[:,:,:,0:length]
    #in_s = functions.upsampling(in_s,real_down.shape[2]*opt.scale_factor,real_down.shape[3]*opt.scale_factor)

    #if n == 0:
    #    in_s = torch.full([real_down.shape[0],real_down.shape[1],real_down.shape[2]*opt.scale_factor,real_down.shape[3]*opt.scale_factor], 0, device=opt.device)
    #else:
    #    in_s = functions.upsampling(real_down,real_down.shape[2]*opt.scale_factor,real_down.shape[3]*opt.scale_factor)


    opt.out_ = '%s/paint2im/%s/%d' % (opt.out_new,n_im,n)
    try:
        os.makedirs('%s' % opt.out_ )
    except OSError:
        pass


    #vutils.save_image(functions.denorm(in_s),
    #                    '%s/in_s.png' %  (opt.out_),
    #                    normalize=False)
    plt.imsave('%s/in_s.png' %  (opt.out_),
                    functions.convert_image_np(in_s), vmin=0, vmax=1)
    plt.imsave('%s/in.png' %  (opt.out_),
                    functions.convert_image_np(paint), vmin=0, vmax=1)

    #vutils.save_image(functions.denorm(real_down),
    #                    '%s/in_down.png' %  (opt.out_),
    #                    normalize=False)
    #vutils.save_image(functions.denorm(real),
    #                    '%s/in.png' %  (opt.out_),
    #                    normalize=False)

    out = retarget2(Gs_,Zs_,reals,NoiseAmp_,opt,in_s,retarget_factor_x,retarget_factor_y,n)
    plt.imsave('%s/Out.png' %  (opt.out_),
                    functions.convert_image_np(out*mask+(1-mask)*reals[-1]))
#del netD
#del netG

'''

### main retargeting###
'''
fold = os.listdir('Input/New3/')
opt = get_arguments()
opt.scale_factor_init = opt.scale_factor
for i in range(len(fold)):
    opt.ref_image = 'New3/%s' % fold[i]
    #opt.paint = 'paint/view3.png'
    image_name = '%s' % fold[i]
    real_ = read_image2np(opt)
    opt.num_scales = int((math.log(math.pow(opt.min_size/(real_.shape[0]),1),opt.scale_factor_init)))+1
    opt.out_ = '%s/%s/%f/%d' % (opt.out,image_name[:-4],opt.small_size,opt.num_scales)
    scale2stop = int(math.log(min([250,max([real_.shape[0],real_.shape[1]])])/max([real_.shape[0],real_.shape[1]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = imresize2.imresize2(real_,scale_factor=opt.scale1, kernel=None, antialiasing=True, kernel_shift_flag=False)
    opt.scale_factor = math.pow(opt.min_size/(real.shape[0]),1/(opt.stop_scale))
    scale2stop = int(math.log(min([250,max([real_.shape[0],real_.shape[1]])])/max([real_.shape[0],real_.shape[1]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop

    Gs = []
    Zs = []
    NoiseAmp = []

    #for i in range(opt.num_scales-1):
    #    opt.nfc_init = 16
    #    opt.nfc = opt.nfc_init*pow(2,i)
    #    opt.outf = '%s/%d' % (opt.out_,i)
    #    netD,netG = init_models(opt)
    #    netG.load_state_dict(torch.load('%s/netG.pth' % opt.outf))
    #    netG.eval()
    #    Gs.append(netG)
    #    z_opt=torch.load('%s/z_opt.pth' % opt.outf)
    #    Zs.append(z_opt)
    #    noise_amp = opt.noise_amp/pow(pow(1/opt.scale_factor,0.5),i)
    #    NoiseAmp.append(noise_amp)

    Gs = torch.load('%s/Gs.pth' % opt.out_)
    Zs = torch.load('%s/Zs.pth' % opt.out_)
    reals = torch.load('%s/reals.pth' % opt.out_)
    NoiseAmp = torch.load('%s/NoiseAmp.pth' % opt.out_)
    reals = torch.load('%s/reals.pth' % opt.out_)

    n = 0
    Gs = Gs[n:]
    Zs = Zs[n:]
    NoiseAmp = NoiseAmp[n:]
    #reals = reals[n:]
    real = reals[n]

    #real = read_image(opt)
    #real = torch.cat((real[:,:,:,int(real.shape[3]/2+1):real.shape[3]],real[:,:,:,1:int(real.shape[3]/2)]),3)

    #paint = img.imread('Input/balloons2/3balloons.png')
    #paint = paint[:,:,:,None]
    #paint = paint.transpose((3, 2, 0, 1))/255
    #paint = torch.from_numpy(paint)
    #paint = functions.move_to_gpu(paint)
    #paint = paint.type(torch.cuda.FloatTensor)
    #paint = functions.norm(paint)

    retarget_factor_x = 0.3
    retarget_factor_y = 1

    #scale = pow(opt.scale_factor,opt.num_scales-1-n)
    #real_down = functions.upsampling(real,real.shape[2],retarget_factor*real.shape[3])
    real_down = functions.upsampling(real,retarget_factor_y*real.shape[2],retarget_factor_x*real.shape[3])

    #in_s = functions.upsampling(paint,reals[n].shape[2],reals[n].shape[3])
    #in_s = functions.upsampling(in_s,real_down.shape[2],real_down.shape[3])
    #in_s = functions.upsampling(real_down,real_down.shape[2]*opt.scale_factor,real_down.shape[3]*opt.scale_factor)
    #in_s = functions.upsampling(in_s,real_down.shape[2],real_down.shape[3])
    #in_s = torch.full(real_down.shape, 0, device=opt.device)

    #in_s = real_down.clone()
    #length = int(real_down.shape[3]/2)-1
    #in_s[:,:,:,0:length] = real_down[:,:,:,int(real_down.shape[3]/2):int(real_down.shape[3]/2)+length]
    #in_s[:,:,:,length:2*length] = real_down[:,:,:,0:length]
    #in_s = functions.upsampling(in_s,real_down.shape[2]*opt.scale_factor,real_down.shape[3]*opt.scale_factor)

    if n == 0:
        in_s = torch.full([real_down.shape[0],real_down.shape[1],real_down.shape[2],real_down.shape[3]], 0, device=opt.device)
    else:
        in_s = functions.upsampling(real_down,real_down.shape[2],real_down.shape[3])



    opt.out_ = '%s/retarget/%d_%f_%f' % (opt.out_,n,retarget_factor_x,retarget_factor_y)
    try:
        os.makedirs('%s' % opt.out_ )
    except OSError:
        pass

    vutils.save_image(functions.denorm(in_s),
                        '%s/in_s.png' %  (opt.out_),
                        normalize=False)
    vutils.save_image(functions.denorm(real_down),
                        '%s/in_down.png' %  (opt.out_),
                        normalize=False)
    vutils.save_image(functions.denorm(real),
                        '%s/in.png' %  (opt.out_),
                        normalize=False)

    retarget2(Gs,Zs,reals,NoiseAmp,opt,in_s,retarget_factor_x,retarget_factor_y,n)
#del netD
#del netG
'''


