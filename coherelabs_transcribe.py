from transformers import AutoProcessor, CohereAsrForConditionalGeneration, pipeline
from transformers.utils import logging as hf_logging
from dotenv import load_dotenv
from huggingface_hub import login
import torch
import soundfile as sf
import soxr
import os
import logging
import argparse
import segment_wav as sw

# Create a generator to load and yield chunks of audio to protect memory
def stream_audio_batches(audio_list, batch_size):
    # Yields small, isolated chunks of raw audio arrays.
    for i in range(0, len(audio_list), batch_size):
        yield audio_list[i : i + batch_size]

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
    # Suppress warnings, only set to print errors
    logger = logging.getLogger("transformers.generation.utils")
    logger.setLevel(logging.ERROR)
    hf_logging.set_verbosity_error()
    # Parse Cmd line arguments
    script_name = os.path.basename(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument('transcript_dir', type=str,
                        help="Absolute or Relative file path to the .csv transcription file with column 1 " \
                        "containing audio PATH and column 2 containing the transcription.")
    # Optional Arg for task 1 and testing purposes
    parser.add_argument('--data_dir', type=str, nargs='?', default = None
                        help="Absolute or Relative file PATH to the .wav file.")
    args = parser.parse_args()
    audio_files = []
    transcriptions, segm_wav, task = process_input(args.transcript_dir, args.data_dir)
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

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "CohereLabs/cohere-transcribe-03-2026"
    model = CohereAsrForConditionalGeneration.from_pretrained(model_id, device_map=device,torch_dtype=torch_dtype)
    processor = AutoProcessor.from_pretrained(model_id)
    # pipe = pipeline(
    #     "automatic-speech-recognition",
    #     model=model,
    #     tokenizer=processor.tokenizer,
    #     feature_extractor=processor.feature_extractor,
    #     chunk_length_s=30,
    #     batch_size = 16,
    #     torch_dtype=torch_dtype,
    #     device=device
    # )
    # audio_stream = (audio for audio in audio_files)
    # results = pipe(audio_stream, batch_size=16, 
    #     generate_kwargs={"language":"en"})
    BATCH_SIZE = 16
    predictions = []
    for batch_idx, batch_audios in enumerate(stream_audio_batches(audio_files, BATCH_SIZE)):
        inputs = processor(batch_audios, sampling_rate=16000, return_tensors="pt", language="en")
        audio_chunk_index = inputs.get("audio_chunk_index")
        inputs.to(model.device, dtype=model.dtype)

        outputs = model.generate(**inputs, max_new_tokens=256)
        results = processor.decode(
            outputs, skip_special_tokens=True, audio_chunk_index=audio_chunk_index, language="en"
        )
        predictions.extend(results)

    results_list = list(predictions)
    print(f"Number of items returned by pipeline: {len(results_list)}")
    # for p, item in enumerate(results):
    #     predictions.append(item['text'])
    sw.workbook_write(transcriptions,predictions, task, script_name)
    if task == 3:
        sw.evaluation(predictions,transcriptions)
    

if __name__ == '__main__':
    main()
