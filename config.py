import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--mode', help='task to be done', default='train')
    #workspace:
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)
    
    #load, input, save configurations:
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    #Audio noise channles will be the same as audio channels
    parser.add_argument('--nc_z',type=int,help='noise # channels',default=1)
    #At least initally we'll be working with 1 channel audio
    parser.add_argument('--nc_aud',type=int,help='image # channels',default=1)
    parser.add_argument('--out',help='output folder',default='Output')
        
    #networks hyper parameters:
    parser.add_argument('--nfc', type=int, default=32)
    #parser.add_argument('--batch_norm', action='store_true', help='use batch normalization (not yet implemented)', default=0)
    parser.add_argument('--batch_norm', type=int, help='use batch normalization (not yet implemented)', default=0)
    parser.add_argument('--min_nfc', type=int, default=32)
    # SinGAN's 3x3 Kernel is flattened to a 9 kernel
    parser.add_argument('--ker_size',type=int,help='kernel size',default=9)
    parser.add_argument('--num_layer',type=int,help='number of layers',default=5)
    parser.add_argument('--stride',help='stride',default=1)
    parser.add_argument('--padd_size',type=int,help='net pad size',default=0)#math.floor(opt.ker_size/2)
    parser.add_argument('--dilation', type=int, help='dilation at each layer', default=1)
    parser.add_argument('--RELU_in_gen', type=int, help='Use RELU instead of leaky RELU in the generator', default=0)


    #pyramid parameters:
    parser.add_argument('--scale_factor',type=float,help='pyramid scale factor',default=0.75)#pow(0.5,1/6))
    parser.add_argument('--noise_amp',type=float,help='addative noise cont weight',default=0.1)
    parser.add_argument('--min_size',type=int,help='image minimal size at the coarser scale',default=25)
    parser.add_argument('--max_size', type=int,help='image minimal size at the coarser scale', default=250)

    #optimization hyper parameters:
    parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train per scale')
    parser.add_argument('--gamma',type=float,help='scheduler gamma',default=0.1)
    parser.add_argument('--lr_g', type=float, default=0.0001, help='learning rate, default=0.0005')
    parser.add_argument('--lr_d', type=float, default=0.0004, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--steps',type=int, help='Generator inner steps',default=3)
    parser.add_argument('--lambda_grad',type=float, help='gradient penelty weight',default=0.1)
    parser.add_argument('--alpha',type=float, help='reconstruction loss weight',default=4)
    parser.add_argument('--dropout', type=float, help='dropout for discriminator', default=0)

    #added by Levi
    #use soundfile for windows and sox io for linux / google colab
    parser.add_argument('--audio_backend', help='normalize audio input?', default='sox_io')
    parser.add_argument('--norm', help='normalize audio input?', default=1)
    parser.add_argument('--wandb', help='log stuff with wandb, you need to init before hand', default=0)
    parser.add_argument('--wandb_proj', help='wandb Project name', default='AudioSinGAN')
    parser.add_argument('--update_every_x', help='only update discriminator every x steps', default=0)
    parser.add_argument('--steps_to_update', help='how many steps to update on', default=10)
    parser.add_argument('--normalize_generator_output', type = int, help='In this experiment I normalize the output of the '
                                                             'generator (before loss is calculated) ', default=0)
    parser.add_argument('--normalize_before_saving', type = int, help='normalize audio output before saving as a file', default=1)
    parser.add_argument('--use_MAE', type = int, help='use MAE instead of LSE', default=0)
    parser.add_argument('--update_only_with_lower_Gloss', type = int, help='update discriminator only when GLoss is '
                                                                           'decreasing', default=0)
    parser.add_argument('--use_schedulers', type = int, help='use DScheduler to decrease lr after 1600 epochs', default=0)
    parser.add_argument('--make_input_tensor_even', type=int, help='make input tensor even', default=1)
    parser.add_argument('--adjust_upsampled', type=int, help='adjust generated tensor size as they are up-sampled', default=1)
    parser.add_argument('--change_channel_count', type=int, help='at higher levels of GAN pyramid use more channels', default=0)
    parser.add_argument('--adjust_after_levels', type=int, help='adjust lr d after level', default=0)
    parser.add_argument('--level_to_resume_at', type=int, help='level to resume training at', default=1)
    parser.add_argument('--alt_pyramid_exp', type=int, help='instead of resampling audio, change kernel size at each '
                                                            'layer of the pyramid', default=0)
    parser.add_argument('--pad_with_noise', type=int, help='pad with noise instead of 0s', default=0)
    parser.add_argument('--single_level', type=int, help='use only a single layer of size x', default=-1)
    parser.add_argument('--save_fake_progression', type=int, help='use this to save a copy of the fake output every 500 epochs', default=-1)
    parser.add_argument('--smooth_real_labels', type=int, help='smooth the real labels of the discriminator', default=-1)
    parser.add_argument('--smooth_fake_labels', type=int, help='smooth the fake labels of the discriminator', default=-1)


    
    return parser
