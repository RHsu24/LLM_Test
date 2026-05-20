import torch
import os
import argparse
import soundfile as sf
import soxr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import segment_wav as sw

# Create a generator to load and yield chunks of audio to protect memory
def stream_audio_batches(audio_list, batch_size):
    # Yields small, isolated chunks of raw audio arrays.
    for i in range(0, len(audio_list), batch_size):
        yield audio_list[i : i + batch_size]


def format_audio(segm_wav):
    audio_files = []
    # Load audio
    for files in segm_wav:
        files.seek(0)
        wave,sr = sf.read(files)
        if sr != 16000:
            wave = soxr.resample(wave, sr, 16000)

        # Convert to mono if it has 2 channels
        waveform = torch.from_numpy(wave).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T
            
        audio_files.append(waveform.squeeze().numpy())
    return audio_files

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
    parser.add_argument('--data_dir', type=str, nargs='?', default = None,
                        help="Absolute or Relative file PATH to the .wav file.")
    args = parser.parse_args()
    transcriptions, segm_wav, task = process_input(args.transcript_dir, args.data_dir)
    audio_files = format_audio(segm_wav)


    ####### MODEL SPECIFIC CODE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "ibm-granite/granite-speech-4.1-2b"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, device_map=device, torch_dtype=torch_dtype
    )
    

    # Create text prompt
    user_prompt = "<|audio|>Transcribe the audio into a written format, assuming Australian English"
    chat = [
        {"role": "user", "content": user_prompt}
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt = True)
    prompts = [prompt] * len(audio_files)

    BATCH_SIZE = 16
    predictions = []
    for batch_idx, batch_audios in enumerate(stream_audio_batches(audio_files, BATCH_SIZE)):
        # Run the processor + model
        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + len(batch_audios) # Protects against the final uneven batch slice
        batch_prompts = prompts[start_idx : end_idx]
        model_inputs = processor(text=batch_prompts, audio=batch_audios, 
                                device=device, return_tensors="pt", padding = True).to(device)
        model_outputs = model.generate(
            **model_inputs, max_new_tokens=256, do_sample=False, num_beams=1
        )

        # Transformers includes the input IDs in the response
        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = model_outputs[:, num_input_tokens:]
        results = tokenizer.batch_decode(
            new_tokens, add_special_tokens=False, skip_special_tokens=True
        )

        predictions.extend(results)
    sw.workbook_write(transcriptions,predictions, task, script_name)
    if task == 3: 
        sw.evaluation(predictions,transcriptions)
    
    print("Finished!!")

if __name__ == '__main__':

    welcome_msg = "This is here to show it is running\n"
    print(welcome_msg)
    
    main()
