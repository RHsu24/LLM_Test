import torch
import os
import soundfile as sf
import soxr
import argparse
import segment_wav as sw
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm

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

def process_input(transcript_dir, data_dir=None):
        # Transcript PATH is not optional
    if transcript_dir is not None:
        if not os.path.exists(transcript_dir):
            raise FileNotFoundError(f'Please input valid .csv file PATH before running the following code.')
        path_ts = transcript_dir
        word_ts = sw.read_transcript(path_ts)
        transcriptions = word_ts
    else:
        raise FileNotFoundError(f'Please input transcription file PATH before running the following code.')
    
    # Existence of Optional Arg {Data PATH} determines if Task 1 (Words) or Task 3 (Sentences) evaluation
    if data_dir is not None:
        if not os.path.exists(data_dir):
               raise FileNotFoundError(f'Please prepare the audio file before running the following code.')
        # For Task 1 - Evaluating Single Words
        task = 1
        path_audio = data_dir
        segm_wav = sw.wav_manip(path_audio,word_ts)

    # If no data_dir argument given, assume .csv file follows format of col 1 - audio path & col 2 - transcript
    else:
        transcriptions = []
        task = 3
        # For Task 3 - Sentences (Assumed .csv format)
        for word in word_ts:
            transcriptions.append(word['Text'])
        segm_wav = sw.wav_manip_long(word_ts)
    return transcriptions, segm_wav, task

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
    transcriptions, segm_wav, task = process_input(args.transcript_dir, args.data_dir)

    audio_files = format_audio(segm_wav)

    ############# MODEL SPECIFIC CODE #####################


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"	
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size = 16,
        torch_dtype=torch_dtype,
        device=device,
    )
    audio_stream = (audio for audio in audio_files)
    results = pipe(audio_stream, batch_size=16, generate_kwargs={"language": "english"})
    results_list = list(results)
    print(f"Number of items returned by pipeline: {len(results_list)}")
    predictions = []
    for p, item in enumerate(results):
        predictions.append(item['text'])

    sw.workbook_write(transcriptions,predictions, task, script_name)
    if task == 3:
        sw.evaluation(predictions,transcriptions)

if __name__ == '__main__':

    welcome_msg = "This is here to show it is running\n"
    print(welcome_msg)
    
    main()
