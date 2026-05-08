import torch
import os
import argparse
import sys
import soundfile as sf
import soxr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import segment_wav as sw

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
    
    if not(os.path.exists(args.data_dir) or os.path.exists(args.transcript_dir)):
        sys.exit("Error: Could not find files on path. Ensure file paths are correct and are either both relative OR both absolute filepaths")
    word_ts = sw.read_transcript(path_ts)
    segm_wav = sw.wav_manip(path_audio,word_ts)
    task = 1

    # Load audio
    audio_files = [] # list of ndarrays to pass into processor
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
            
        audio_files.append(waveform.squeeze().numpy())
            

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "ibm-granite/granite-speech-4.1-2b"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.bfloat32
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
    # Run the processor + model
    model_inputs = processor(text=prompts, audio=audio_files, 
                             device=device, return_tensors="pt", padding = True).to(device)
    model_outputs = model.generate(
        **model_inputs, max_new_tokens=256, do_sample=False, num_beams=1
    )

    # Transformers includes the input IDs in the response
    num_input_tokens = model_inputs["input_ids"].shape[-1]
    new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
    results = tokenizer.batch_decode(
        model_outputs, add_special_tokens=False, skip_special_tokens=True
    )
    recognised_word = []
    for rec_word in results:
    # The results list containts user prompt & assistant reply, so we 
    # will need to process it ourselves
        reply = rec_word.rpartition("ASSISTANT:")
        recognised_word.append(reply[2])
        print(reply[2])
    sw.workbook_write(word_ts,recognised_word, task, script_name)
    print("Finished!!")

if __name__ == '__main__':

    welcome_msg = "This is here to show it is running\n"
    print(welcome_msg)
    
    main()
