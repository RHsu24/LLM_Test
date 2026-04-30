from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio
from dotenv import load_dotenv
from huggingface_hub import login
import torch
import torchaudio
import os
from pathlib import Path
import argparse
import glob
import xlsxwriter
import sys
import segment_wav as sw

def simple_split(namelist, slice):
    
    word_extr = []
    rubbish = namelist.rpartition("_")
    if slice == 0: #cuts out the number at the end
        word_extr = rubbish[0]
    if slice == 1: #keeps word_id as of files
        word_id = rubbish[2].split(".")
        word_extr = word_id[0]
    return word_extr

def workbook_write(word_files, results):
    wb_name = 'coherelabstranscribe_test.xlsx'
    workbook = xlsxwriter.Workbook(wb_name)
    worksheet = workbook.add_worksheet()
    bold = workbook.add_format({'bold':True})
    worksheet.write('A1','Expected', bold)
    worksheet.write('B1','Recognised', bold)
    worksheet.write('C1', "Binary_eval", bold)

    for i, (word,r) in enumerate(zip(word_files,results)):
        read_word = word['text']
        #clean up the words for comparison - combine multi-words into one word 
        word_clean = sw.normalise_word(read_word)
        asr_clean = sw.normalise_word(r)
        worksheet.write(i+1,0,word_clean)
        worksheet.write(i+1,1,asr_clean)
        worksheet.write(i+1,2, word_clean.casefold() == asr_clean.casefold())

    workbook.close()


def main():
    # Parse Cmd line arguments
    cwd = Path.cwd()
    default_path = Path("/srv/scratch/z5207649/1565")
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, default=default_path,
                        help="Absolute or Relative file path to the .wav file.")
    parser.add_argument("transcript_dir", type=str, default=default_path,
                        help="Absolute or Relative file path to the annotated transcription file.")
    args = parser.parse_args()
    path_ts = args.transcript_dir
    path_audio = args.data_dir
    
    if not(os.path.exists(args.data_dir) or os.path.exists(args.transcript_dir)):
        sys.exit("Error: Could not find files on path. Ensure file paths are correct and are either both relative OR both absolute filepaths")
    word_ts = sw.read_transcript(path_ts)
    segm_wav = sw.wav_manip(path_audio,word_ts)

    # Access gated model
    # Load the .env file
    load_dotenv()
    # Get the token and log in
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("Warning: HF_TOKEN not found in environment.")

    # Load audio

    audio_files = [] # list of ndarrays to pass into processor
    for files in segm_wav:
        files.seek(0)
        word_audio,sr = torchaudio.load(files, format="wav")
        #print(files)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(word_audio)
    
        # 4. Convert to mono if it has 2 channels
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        audio_files.append(waveform.squeeze().numpy())

    processor = AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026")
    model = CohereAsrForConditionalGeneration.from_pretrained("CohereLabs/cohere-transcribe-03-2026", device_map="auto")

    inputs = processor(audio_files, sampling_rate=16000, return_tensors="pt", language="en")
    audio_chunk_index = inputs.get("audio_chunk_index")
    inputs.to(model.device, dtype=model.dtype)

    outputs = model.generate(**inputs, max_new_tokens=256)
    results = processor.decode(
        outputs, skip_special_tokens=True, audio_chunk_index=audio_chunk_index, language="en"
    )
    #print(text)
    workbook_write(word_ts,results)
    print("Finished!!")

if __name__ == '__main__':

    welcome_msg = "This is here to show it is running\n"
    print(welcome_msg)
    
    main()
