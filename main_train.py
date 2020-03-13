import shelve

from SinGAN.manipulate import *
from SinGAN.training import *


def read_options():
    opt1 = parser.parse_args()
    opt1 = functions.post_config(opt1)
    try:
        sh = shelve.open('opt')
        opt1 = sh['opt']
        sh.close()
    except KeyError:
        print('Serialized options file not found or is empty')
        pass
    return opt1


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir',
                        default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--scale_num',
                        help='Current scale',
                        type=int, default=0)

    opt = read_options()

    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    print(dir2save)
    if os.path.exists(dir2save):
        print('Dir exists')
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
    real = functions.read_image(opt)
    functions.adjust_scales2image(real, opt)
    train(opt, Gs, Zs, reals, NoiseAmp)
    SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt)
