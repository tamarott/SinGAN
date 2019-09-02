import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--mode', help='task to be done', default='train')
    #workspace:
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--cuda', action='store_true', help='enables cuda', default=1)
    
    #load, input, save configurations:
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nc_z',type=int,help='noise # channels',default=3)
    parser.add_argument('--nc_im',type=int,help='image # channels',default=3)
    parser.add_argument('--out',help='output folder',default='Output')
        
    #networks hyper parameters:
    parser.add_argument('--nfc', type=int, default=32)
    parser.add_argument('--min_nfc', type=int, default=32)
    parser.add_argument('--ker_size',type=int,help='kernel size',default=3)
    parser.add_argument('--num_layer',type=int,help='number of layers',default=5)
    parser.add_argument('--stride',help='stride',default=1)
    parser.add_argument('--padd_size',type=int,help='net pad size',default=0)#math.floor(opt.ker_size/2)
        
    #pyramid parameters:
    parser.add_argument('--scale_factor',type=float,help='pyramid scale factor',default=0.75)#pow(0.5,1/6))
    parser.add_argument('--noise_amp',type=float,help='addative noise cont weight',default=0.1)
    parser.add_argument('--min_size',type=int,help='image minimal size at the coarser scale',default=25)
    parser.add_argument('--max_size', type=int,help='image minimal size at the coarser scale', default=256)

    #optimization hyper parameters:
    parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train per scale')
    parser.add_argument('--gamma',type=float,help='scheduler gamma',default=0.1)
    parser.add_argument('--lr_g', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--lr_d', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--Gsteps',type=int, help='Generator inner steps',default=3)
    parser.add_argument('--Dsteps',type=int, help='Discriminator inner steps',default=3)
    parser.add_argument('--lambda_grad',type=float, help='gradient penelty weight',default=0.1)
    parser.add_argument('--alpha',type=int, help='reconstruction loss weight',default=10)

    # MANIPULATIONS PARAMETERS
    # random_samples
    # animation
    # SR
    parser.add_argument('--sr_factor', type=float, help='Super resolution factor', default=4)
    #harmonization,editing
    parser.add_argument('--ref_dir', help='input paint dir', default='Input/Edit')
    parser.add_argument('--ref_name', help='ref image name', default='stone_edit.png')
    #parser.add_argument('--mask_name', help='mask image name', default='birds_mask.png')
    parser.add_argument('--editing_start_scale', type=int, help='editing start scale', default=3)
    #parser.add_argument('--harmonization_start_scale', type=int, help='harmonization start scale', default=0)
    #paint2image
    parser.add_argument('--paint_dir', help='input paint dir', default='Input/Paints')
    parser.add_argument('--paint_name', help='paint image name', default='trees3_paint2.png')
    parser.add_argument('--paint_start_scale', type=int, help='paint start scale', default=2)
    parser.add_argument('--quantization_flag', type=bool, help='quantization_flag', default=True)
    parser.add_argument('--quantization_levels', type=int, help='paint quantization levels', default=5)

    return parser