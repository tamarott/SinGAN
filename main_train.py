from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
import torchaudio
import os
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
    parser.add_argument('--input_dir', help='input image dir',
                        default='Input/Audio')  # Changed by Levi Pfantz 10/14/2020
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')


    opt = parser.parse_args()
    opt = functions.post_config(opt)
    opt.batch_norm=bool(opt.batch_norm)
    opt.SR_pyr=[800, 1600, 2150, 2850, 3825, 5100, 6750, 9000, 12000, 16000]

    opt.ker_size_pyr = [1205, 3015, 401, 401, 401, 401, 401, 401, 401, 401, 401, 401, 401, 401, 301, 201]

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
    train(opt, real, Gs, Zs, reals, NoiseAmp)
    SinGAN_generate_audio(Gs,Zs,reals,NoiseAmp,opt)
    my_Logger.close()


