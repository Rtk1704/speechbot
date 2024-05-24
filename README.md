As far as I understand, examples/textless_nlp/gslm/tools/resynthesize_speech.py should take a speech sample (audio), encode it to units, and generate output speech from these units. The output speech should resemble the input sample.

However, when I do this with the released pre-trained models, output is gibberish that doesn't sound like input at all.

I attach the samples and steps I took. Is there anything I do is wrong?

Thank you!

Code
Download pre-trained models (HuBERT-km200 in this example):
mkdir -p /content/speech/hubert200
cd /content/speech/hubert200
wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt -nc 
wget https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km200/km.bin -nc
wget https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/tts_km200/tts_checkpoint_best.pt -nc 
wget https://dl.fbaipublicfiles.com/textless_nlp/gslm/waveglow_256channels_new.pt -nc
Generate the code_dict.txt file. I didn't find "official" description of how to do it, so I used this comment. Note that if I use dict of size 199 or 200, the models will fail
with open("code_dict.txt", "wt") as f:
    for i in range(1, 199):   # Effectively 198 items
        f.write(str(i) + "\n")
Download and convert source audio sample from the speech resynthesis example site:
wget https://speechbot.github.io/resynthesis/audio/teaser/p269_182.mp3 -nc
ffmpeg -y -i p269_182.mp3 sample.input.wav
Run resynthesis:
export FAIRSEQ_ROOT=/home/ubuntu/fairseq
export DATA=/content/speech/hubert200
export TYPE=hubert

echo sample.input.wav > input.txt
echo sample.out.layer5.wav >> input.txt

PYTHONPATH=${FAIRSEQ_ROOT}:${FAIRSEQ_ROOT}/examples/textless_nlp/gslm/unit2speech python ${FAIRSEQ_ROOT}/examples/textless_nlp/gslm/tools/resynthesize_speech.py \
    --feature_type $TYPE \
    --layer 5 \
    --acoustic_model_path $DATA/hubert_base_ls960.pt \
    --kmeans_model_path $DATA/km.bin \
    --tts_model_path $DATA/tts_checkpoint_best.pt \
    --code_dict_path $DATA/code_dict.txt \
    --waveglow_path $DATA/waveglow_256channels_new.pt \
    --max_decoder_steps 1000 < input.txt
Check the result (in the attachement
). It doesn't sound like the original audio at all.
What have you tried?
I tried to run resynthesis with different number of units, taking different HuBERT layer for features, different audio, and different offsets for code_dict.txt

In addition to steps outlined above, I tried to generate speech with units2speech directly from units in devset. It still produces gibberish. This makes me think that the problem may lie in bad pre-trained tts checkpoint.

What's your environment?
fairseq Version (e.g., 1.0 or main): main
PyTorch Version (e.g., 1.0) 1.9.1
OS (e.g., Linux): Ubuntu 18.04
How you installed fairseq (pip, source): source
Build command you used (if compiling from source): pip install -e .
Python version: 3.7.0
CUDA/cuDNN version: cuda_11.1.TC455_06.29190527_0
GPU models and configuration: Tesla V100-SXM2
Any other relevant information:
samples.zip contains generates samples - both audio and units.

@osyvokon osyvokon added needs triage question labels on Oct 21, 2021
@osyvokon osyvokon mentioned this issue on Oct 21, 2021
Textless NLP. Missed code_dict data #3870
Closed
@orcsun
orcsun commented on Oct 21, 2021
thank you asivokon for posting this issue. i ran into the same.
seems like missing code_dict.txt file is doing some magic mappings which should not just be a seq of index from 1 to N.

@Uncomfy
Uncomfy commented on Oct 21, 2021
Most of the TTS models I've tried have smaller embedding layer than the amount of units in respective K-means models (which is strange, since TTS adds one symbol for padding and one for EOS, so it should be bigger). One of the models that has same sizes in K-means and TTS embeddings is HuBERT + KM50, but it still produces gibberish for me if I use dictionary with numbers from 1 to 50 or 0 to 49.

@osyvokon
Contributor
Author
osyvokon commented on Oct 27, 2021
@hikushalhere, could you please help with this?

@eugene-kharitonov
Contributor
eugene-kharitonov commented on Oct 27, 2021
Hello, @asivokon ,
Thanks for your interest to our work!

Please use Hubert layer 6, I believe the pre-trained checkpoints assume this.
I can confirm that code_dict for eg Hubert100 is just a text file with numbers [0...99] inclusive.
[Here] I've included a code file and a resynthesis output for your input example obtained using vocab-100 model.
Please let me know if this solves your issue.

@bradgrimm
bradgrimm commented on Oct 28, 2021
@eugene-kharitonov No luck for me. If I use your code_dict it crashes.

RuntimeError: cuDNN error: CUDNN_STATUS_MAPPING_ERROR
From what I can tell the hubert km 100 file has n_symbols=101. Which means 99 numbers (0-98) plus the pad token and the end of sentence token. If I remove one number (either from the beginning or end) it does not crash anymore, but I get gibberish.

To make it even more confusing the cpc file has n_symbols=102. So it has one more symbol than hubert (and should work with your file), but your file also has an empty newline at the end causing it to crash (not sure if that is intentional). If I remove the newline it stops crashing, I get gibberish with the CPC models too.

Are you certain the uploaded files are correct? I'm downloading the following files:
Acoustic: https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt
SPU: https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km100/km.bin
UTS: https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/tts_km100/tts_checkpoint_best.pt
Waveglow: https://dl.fbaipublicfiles.com/textless_nlp/gslm/waveglow_256channels_new.pt

Here's the exact command I'm using:

PYTHONPATH=${FAIRSEQ_ROOT}:${FAIRSEQ_ROOT}/examples/textless_nlp/gslm/unit2speech python ${FAIRSEQ_ROOT}/examples/textless_nlp/gslm/tools/resynthesize_speech.py \
    --feature_type hubert \
    --acoustic_model_path /mnt/large/data/pretrained/hubert_base_ls960.pt \
    --layer 6 \
    --kmeans_model_path /mnt/large/data/pretrained/km_100.bin \
    --tts_model_path /mnt/large/data/pretrained/hubert_base_km100.pt \
    --code_dict_path /mnt/large/data/pretrained/code_dict_100 \
    --waveglow_path /mnt/large/data/pretrained/waveglow_256channels_new.pt \
    --max_decoder_steps 2000
