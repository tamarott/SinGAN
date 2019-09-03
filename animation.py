from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    #parser.add_argument('--animation_start_scale', type=int, help='generation start scale', default=2)
    parser.add_argument('--alpha_animation', type=float, help='animation random walk first moment', default=0.1)
    #parser.add_argument('--beta_animation', type=float, help='animation random walk second moment', default=0.9)
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='animation')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    if (os.path.exists(dir2save)):
        print("output already exist")
    else:
        opt.min_size = 20
        opt.mode = 'animation_train'
        real = functions.read_image(opt)
        functions.adjust_scales2image(real, opt)
        dir2trained_model = functions.generate_dir2save(opt)
        if (os.path.exists(dir2trained_model)):
            Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
            opt.mode = 'animation'
        else:
            train(opt, Gs, Zs, reals, NoiseAmp)
            opt.mode = 'animation'
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        for start_scale in range(0, 3, 1):
            for b in range(80, 100, 5):
                #opt.animation_start_scale = start_scale
                #opt.beta_animation = b / 100
                generate_gif(Gs, Zs, reals, NoiseAmp, opt, beta=b/100, start_scale=start_scale)

