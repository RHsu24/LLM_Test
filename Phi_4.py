import os
import torch
import soundfile as sf
import argparse
import segment_wav as sw
import soxr
import logging
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

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
            
        audio_files.append((waveform.squeeze().numpy(), 16000))
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
    # Suppress warnings, only set to print errors
    logger = logging.getLogger("transformers.generation.utils")
    logger.setLevel(logging.ERROR)
    # Process input
    script_name = os.path.basename(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument('transcript_dir', type=str,
                        help="Absolute or Relative file path to the .csv transcription file with column 1 " \
                        "containing audio PATH and column 2 containing the transcription.")
    # Optional Arg for task 1 and testing purposes
    parser.add_argument('--data_dir', type=str, nargs='?', default=None,
                        help="Absolute or Relative file PATH to the .wav file.")
    args = parser.parse_args()
    transcriptions, segm_wav, task = process_input(args.transcript_dir, args.data_dir)

    audio_files = format_audio(segm_wav)

    #################### Model-Specific Code ################
    model_path = "Lexius/Phi-4-multimodal-instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {}
    kwargs['torch_dtype'] = torch.bfloat16

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map = None,
        trust_remote_code=True,
        torch_dtype='auto',
        _attn_implementation='sdpa',
    ).to(device)
    # print("model.config._attn_implementation:", model.config._attn_implementation)
    generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')

    user_prompt = "<|audio_1|>Transcribe the audio into a written format, assuming Australian English"
    batched_chat = [
        [{"role": "user", "content": user_prompt}]
	    for _ in range(len(audio_files))
    ]
    prompts = [
        processor.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt = True)
        for msg in batched_chat
    ]

    BATCH_SIZE = 16
    predictions = []
    for batch_idx, batch_audios in enumerate(stream_audio_batches(audio_files, BATCH_SIZE)):
        # Run the processor + model
        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + len(batch_audios) # Protects against the final uneven batch slice
        batch_prompts = prompts[start_idx : end_idx]
        inputs = processor(text=batch_prompts, audios=batch_audios, padding=True, return_tensors='pt').to('cuda:0')

        generate_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            generation_config=generation_config,
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        predictions.extend(response)

    sw.workbook_write(transcriptions,predictions, task, script_name)
    if task == 3:
        sw.evaluation(predictions,transcriptions)
if __name__ == '__main__':
    main()
