import os
import torch
import soundfile as sf
import argparse
import segment_wav as sw
import soxr
import sys
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

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

def main():
    # Process input
    script_name = os.path.basename(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument('transcript_dir', type=str,
                        help="Absolute or Relative file path to the .csv transcription file with column 1 " \
                        "containing audio PATH and column 2 containing the transcription.")
    # Optional Arg for task 1 and testing purposes
    parser.add_argument('--data_dir', type=str, nargs='?',
                        help="Absolute or Relative file PATH to the .wav file.")
    args = parser.parse_args()
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
        for word in word_ts:
            transcriptions.append(word['Text'])
        segm_wav = sw.wav_manip_long(word_ts)
        
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
        _attn_implementation='eager',
    ).cuda()
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

    formatted_inputs = []
    for buf in audio_files:
        formatted_inputs.append((buf,16000))

    try:
        inputs = processor(text=prompts, audios=formatted_inputs, padding=True, return_tensors='pt').to('cuda:0')
    except AssertionError as e:
        print(f"Crash detected: {e}")
        # Manually tokenize to see what's happening
        test_ids = processor.tokenizer(prompts[0])["input_ids"]
        decoded = [processor.tokenizer.decode([i]) for i in test_ids]
        print(f"How the tokenizer saw your prompt: {decoded}")
        raise e
    
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        generation_config=generation_config,
    )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    if task == 1:
        sw.workbook_write(word_ts,response, task, script_name)
    else:
        sw.evaluation(response,transcriptions)
if __name__ == '__main__':
    main()
