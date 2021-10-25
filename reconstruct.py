from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from utils.config_manager import Config
from model.factory import tts_ljspeech, tts_custom
from data.audio import Audio

if __name__ == '__main__':
    defaultConf = "config/peeter_jutustav_16/session_paths.yaml"

    parser = ArgumentParser()
    parser.add_argument('--config', '-c', dest='config', default=defaultConf, type=str)
    parser.add_argument('--text', '-t', dest='text', default=None, type=str)
    parser.add_argument('--file', '-f', dest='file', default=None, type=str)
    parser.add_argument('--weights', '-w', dest='weights', default=None, type=str)
    parser.add_argument('--checkpoint', '-ckpt', dest='checkpoint', default=None, type=str)
    parser.add_argument('--outdir', '-o', dest='outdir', default=None, type=str)
    parser.add_argument('--store_mel', '-m', dest='store_mel', action='store_true')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true')
    parser.add_argument('--single', '-s', dest='single', action='store_true')
    args = parser.parse_args()
    
    if args.file is not None:
        with open(args.file, 'r', encoding='utf-8') as file:
            text = file.readlines()
        fname = Path(args.file).stem
    elif args.text is not None:
        text = [args.text]
        fname = 'custom_text'
    else:
        fname = None
        text = None
        #print(f'Specify either an input text (-t "some text") or a text input file (-f /path/to/file.txt)')
        #exit()
    # load the appropriate model
    outdir = Path(args.outdir) if args.outdir is not None else Path('.')
    if args.config is not None:
        if args.weights is not None:
            model, conf = tts_custom(args.config, args.weights)
            file_name = f'{fname}_{Path(args.weights).stem}'
        else:
            config_loader = Config(config_path=args.config)
            outdir = Path(args.outdir) if args.outdir is not None else Path(config_loader.log_dir)
            conf = config_loader.config
            model = config_loader.load_model(args.checkpoint)  # if None defaults to latest
            file_name = f'{fname}_tts_step{model.step}'
    else:
        model, conf = tts_ljspeech()
        file_name = f'{fname}_ljspeech_v1'
    
    outdir = outdir  / f'{fname}'
    outdir.mkdir(exist_ok=True, parents=True)
    audio = Audio(conf)
    print(f'Output wav under {outdir}')
    #csv_path = '/home/egert/Prog/TransformerTTS/testout/custom_text/custom_text_tts_step260000_0.csv'
    csv_path = '/home/egert/Prog/TTS-CPP/TransformerTTS-Cpp/src/test.csv'
    mel = np.loadtxt(csv_path,
                     delimiter=',')
    mel = np.load("/home/egert/Prog/TransformerTTS/testout/custom_text/custom_text_tts_step260000_0.mel.npy")
    
    print(mel.shape)
    wav = audio.reconstruct_waveform(mel.T)
    audio.save_wav(wav, '/home/egert/Prog/TTS-CPP/TransformerTTS-Cpp/src/test.wav')
