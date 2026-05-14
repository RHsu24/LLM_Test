import torch
from qwen_asr import Qwen3ASRModel
import os
import numpy as np
import soundfile as sf
import soxr
import argparse
import sys
import segment_wav as sw

def format_audio(segm_wav):
    formatted_audio = []
# Load audio into correct format for ASR model
    for files in segm_wav:
        files.seek(0)
        word_audio,sr = sf.read(files)
        if sr != 16000:
            wave = soxr.resample(word_audio, sr, 16000)

        # Convert to mono if it has 2 channels
        waveform = torch.from_numpy(wave).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T
            
    formatted_audio.append(waveform.squeeze().numpy())
    return formatted_audio

def main():
    # Parse Cmd line arguments
    os.environ['HF_HOME'] = '/srv/scratch/z5207649/huggingface_cache'
    script_name = os.path.basename(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcript_dir', type=str, required=True,
                        help="Absolute or Relative file path to the .csv transcription file with column 1 " \
                        "containing audio PATH and column 2 containing the transcription.")
    # Optional Arg for task 1 and testing purposes
    parser.add_argument('--data_dir', type=str, nargs='?',
                        help="Absolute or Relative file PATH to the .wav file.")
    args = parser.parse_args()
    path_ts = args.transcript_dir
    path_audio = args.data_dir
    audio_files = []
    # Transcript PATH is not optional
    if args.transcript_dir is not None:
        if not os.path.exists(args.transcript_dir):
            raise FileNotFoundError(f'Please input valid .csv file PATH before running the following code.')
        path_ts = args.transcript_dir
        word_ts = sw.read_transcript(path_ts)
    else:
        raise FileNotFoundError(f'Please input transcription file PATH before running the following code.')
    # Existence of Optional Arg {Data PATH} determines if Task 1 (Words) or Task 3 (Sentences) evaluation
    if args.data_dir is not None:
        if not os.path.exists(args.data_dir):
               raise FileNotFoundError(f'Please prepare the audio file before running the following code.')
        # For Task 1 - Evaluating Single Words
        task = 1
        path_audio = args.data_dir
        segm_wav = sw.wav_manip(path_audio,word_ts)
        audio_files.append(format_audio(segm_wav))

    # If no data_dir argument given, assume .csv file follows format of col 1 - audio path & col 2 - transcript
    else:
        transcriptions = []
        task = 3
        # For Task 3 - Sentences (Assumed .csv format)
        for word in word_ts:
            transcriptions.append(word['Text'])
            
        segm_wav = sw.wav_manip_long(word_ts)
        audio_files = format_audio(segm_wav)


    # Qwen3ASRModel natively converts our 44.1kHZ input to 16kHz
    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map="auto",
        # attn_implementation="flash_attention_2",
        max_inference_batch_size=16, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
        max_new_tokens=16, # Maximum number of tokens to generate. Set a larger value for long audio input.
    )
    formatted_inputs = []
    for buf in audio_files:
        formatted_inputs.append((buf, 16000))
    results = model.transcribe(
        audio=formatted_inputs,
        language="English", # set "English" to force the language
    )
    recognised_text = []
    for rec_word in results:
        recognised_text.append(rec_word.text)
    sw.workbook_write(word_ts,recognised_text, task, script_name)
    print("Finished!!")

if __name__ == '__main__':

    welcome_msg = "This is here to show it is running\n"
    print(welcome_msg)
    
    main()
