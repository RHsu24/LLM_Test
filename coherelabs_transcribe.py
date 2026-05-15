from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from dotenv import load_dotenv
from huggingface_hub import login
import torch
import soundfile as sf
import soxr
import os
import argparse
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
    script_name = os.path.basename(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument('transcript_dir', type=str,
                        help="Absolute or Relative file path to the .csv transcription file with column 1 " \
                        "containing audio PATH and column 2 containing the transcription.")
    # Optional Arg for task 1 and testing purposes
    parser.add_argument('--data_dir', type=str, nargs='?',
                        help="Absolute or Relative file PATH to the .wav file.")
    args = parser.parse_args()
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
        audio_files = format_audio(segm_wav)

    # If no data_dir argument given, assume .csv file follows format of col 1 - audio path & col 2 - transcript
    else:
        transcriptions = []
        task = 3
        # For Task 3 - Sentences (Assumed .csv format)
        for word in word_ts:
            transcriptions.append(word['Text'])
            
        segm_wav = sw.wav_manip_long(word_ts)
        audio_files = format_audio(segm_wav)

    # Access gated model
    # Load the .env file
    load_dotenv()
    # Get the token and log in
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        raise NameError(f'Warning: HF_TOKEN not found in environment.')
    
    processor = AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026")
    model = CohereAsrForConditionalGeneration.from_pretrained("CohereLabs/cohere-transcribe-03-2026", device_map="auto")

    inputs = processor(audio_files, sampling_rate=16000, return_tensors="pt", language="en")
    audio_chunk_index = inputs.get("audio_chunk_index")
    inputs.to(model.device, dtype=model.dtype)

    outputs = model.generate(**inputs, max_new_tokens=256)
    results = processor.decode(
        outputs, skip_special_tokens=True, audio_chunk_index=audio_chunk_index, language="en"
    )
    if task == 1:
        sw.workbook_write(word_ts,results, task, script_name)
    else: 
        sw.evaluation(results,transcriptions)
    

if __name__ == '__main__':
    main()
