from transformers import AutoProcessor, VoxtralForConditionalGeneration, BitsAndBytesConfig
import torch
import os
import librosa
import argparse
import segment_wav as sw

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
    # If no data_dir argument given, assume .csv file follows format of col 1 - audio path & col 2 - transcript
    else:
        transcriptions = []
        task = 3
        # For Task 3 - Sentences (Assumed .csv format)
        for _ , word in enumerate(word_ts):
            transcriptions.append(word['Text'])
        segm_wav = sw.wav_manip_long(word_ts)

    # Load audio
    for files in segm_wav:
        files.seek(0)
        waveform, _ = librosa.load(files,sr=16000)
        audio_files.append(waveform)
    print(f'Has Total {len(audio_files)} audio files')


    device = "cuda"
    repo_id = "mistralai/Voxtral-Small-24B-2507"

    processor = AutoProcessor.from_pretrained(repo_id)
    model = VoxtralForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map='auto',low_cpu_mem_usage=True)

    languages = ["en"] * len(audio_files) 
    model_ids = [repo_id] * len(audio_files)
     
    inputs = processor.apply_transcription_request(language=languages, audio=audio_files, model_id=model_ids) 
    inputs = inputs.to(device, dtype=torch.bfloat16) 

    outputs = model.generate(**inputs, max_new_tokens=500) 
    decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

    if task == 1:
        sw.workbook_write(word_ts,decoded_outputs, task, script_name)
    else: 
        sw.evaluation(decoded_outputs,transcriptions)
    

if __name__ == '__main__':
    main()
