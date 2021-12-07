# I added this file. When I made it I started with main_train as a base
# so they have a lot of overlap.

from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
import torchaudio
import sys
from datetime import datetime
from SinGAN.AudioSample import AudioSample

class my_Logger(object):
    def __init__(self):
        self.console_out = sys.stdout
        self.file = open("logFile.txt", "a")
        self.encoding = "UTF-8"

    def write(self, input):
        self.console_out.write(input)
        self.file.write(input)

    def close(self):
        self.file.close()

    def flush(self):
        pass

if __name__ == '__main__':

    sys.stdout=my_Logger()

    print(str(sys.argv))

    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir',default='Input/Audio')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')

    opt = parser.parse_args()
    opt = functions.post_config(opt)
    opt.SR_pyr=[800, 1600, 2150, 2850, 3825, 5100, 6750, 9000, 12000, 16000]

    if opt.single_level >0:
        opt.SR_pyr=[opt.single_level]


    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)


    if not opt.not_cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    if int(opt.wandb) > 0:
        import wandb
        wandb.init(project=opt.wandb_proj)


    torchaudio.set_audio_backend(opt.audio_backend)
    inputpath = opt.input_dir + "/" + opt.input_name


    real = AudioSample(opt, inputpath, sr=16000)
    opt.stride=int(opt.stride)
    opt.nfc=int(opt.nfc)
    opt.min_nfc=int(opt.min_nfc)
    print(real.data.shape)
    functions.adjust_scales2audio(real, opt)
    opt.stop_scale=len(opt.SR_pyr)-1
    print(real.data.shape)
    Gs, Zs, reals_trash, NoiseAmp = functions.load_trained_pyramid(opt)
    del reals_trash
    reals=[]
    reals = functions.creat_reals_pyramid_julius(real, reals, opt, verbose=False)

    opt.mode = 'train'
    Gs=Gs[0:opt.level_to_resume_at]
    Zs=Zs[0:opt.level_to_resume_at]
    NoiseAmp=NoiseAmp[0:opt.level_to_resume_at]
    in_s = torch.full([1, reals[0].shape[0], reals[0].shape[2]], 0, dtype=torch.float32, device=opt.device)
    if opt.level_to_resume_at < len(opt.SR_pyr):
        train_on_audio_resume(opt, real, Gs, Zs, reals, NoiseAmp, in_s)
    SinGAN_generate_audio(Gs,Zs,reals,NoiseAmp,opt)
    my_Logger.close()
