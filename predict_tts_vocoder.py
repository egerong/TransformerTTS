from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from model.factory import tts_ljspeech, tts_custom
from utils.config_manager import Config
from data.audio import Audio
from vocoding.predictors import HiFiGANPredictor, MelGANPredictor
from utils.scripts_utils import tf_mem_limit

tf_mem_limit(4096)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', dest='config', default=None, type=str)
    parser.add_argument('--text', '-t', dest='text', default=None, type=str)
    parser.add_argument('--file', '-f', dest='file', default=None, type=str)
    parser.add_argument('--weights', '-w', dest='weights', default=None, type=str)
    parser.add_argument('--checkpoint', '-ckpt', dest='checkpoint', default=None, type=str)
    parser.add_argument('--outdir', '-o', dest='outdir', default=None, type=str)
    parser.add_argument('--store_mel', '-m', dest='store_mel', action='store_true')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true')
    parser.add_argument('--single', '-s', dest='single', action='store_true')
    parser.add_argument('--voc', dest='vocoder', default='glim')
    args = parser.parse_args()
    
    if args.file is not None:
        with open(args.file, 'r') as file:
            text = file.readlines()
        fname = Path(args.file).stem
    elif args.text is not None:
        text = [args.text]
        fname = 'custom_text'
    else:
        fname = None
        text = None
        print(f'Specify either an input text (-t "some text") or a text input file (-f /path/to/file.txt)')
        exit()
    # load the appropriate model
    outdir = Path(args.outdir) if args.outdir is not None else Path('.')
    if args.config is not None:
        if args.weights is not None:
            model, conf = tts_custom(args.config, args.weights)
            file_name = f'{fname}_{args.vocoder}_{Path(args.weights).stem}'
        else:
            config_loader = Config(config_path=args.config)
            outdir = Path(args.outdir) if args.outdir is not None else Path(config_loader.log_dir)
            conf = config_loader.config
            model = config_loader.load_model(args.checkpoint)  # if None defaults to latest
            file_name = f'{fname}_{args.vocoder}_tts_step{model.step}'
    else:
        model, conf = tts_ljspeech()
        file_name = f'{fname}_{args.vocoder}_ljspeech_v1'
    
    if args.vocoder == 'melgan':
        vocoder = MelGANPredictor.from_folder('vocoding/melgan/en')
    elif args.vocoder == 'hifigan':
        vocoder = HiFiGANPredictor.from_folder('vocoding/hifigan/en')
    outdir = outdir / 'outputs' / f'{fname}'
    outdir.mkdir(exist_ok=True, parents=True)
    audio = Audio(conf)
    print(f'Output wav under {outdir}')
    wavs = []
    for i, text_line in enumerate(text):
        phons = model.text_pipeline.phonemizer(text_line)
        tokens = model.text_pipeline.tokenizer(phons)
        if args.verbose:
            print(f'Predicting {text_line}')
            print(f'Phonemes: "{phons}"')
            print(f'Tokens: "{tokens}"')
        out = model.predict(tokens, encode=False, phoneme_max_duration=None)
        mel = out['mel'].numpy().T
        if args.vocoder == 'glim':
            wav = audio.reconstruct_waveform(mel)
        else:
            wav = vocoder([mel])[0]
        wavs.append(wav)
        if args.store_mel:
            np.save((outdir / (file_name + f'_{i}')).with_suffix('.mel'), out['mel'].numpy())
        if args.single:
            audio.save_wav(wav, (outdir / (file_name + f'_{i}')).with_suffix('.wav'))
    audio.save_wav(np.concatenate(wavs), (outdir / file_name).with_suffix('.wav'))