from nemo.collections.speechlm2.models import SALM
import torch
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
import soxr
import os
import argparse
import segment_wav as sw

def format_audio(segm_wav):
    audio_frames = []
    audio_files = []
    # Load audio
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
        audio_frames.append(waveform.size)
        audio_files.append(waveform.squeeze().numpy())

    tensor_list = [torch.as_tensor(audio).float().squeeze() for audio in audio_files] # Convert arrays into 1D PyTorch tensors
    # Track original unpadded lengths (for NeMo unmasking)
    audio_lengths = torch.tensor([t.shape[0] for t in tensor_list], dtype=torch.int32)
    # Dynamic padding to match the longest audio track batch, makes rectangular tensor of shape (batch_size, max_sequence_length)
    audios = pad_sequence(tensor_list, batch_first=True, padding_value=0.0)

    return audio_lengths, audios

def main():
    ########## Parse Cmd line arguments ############
    script_name = os.path.basename(__file__)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
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
        audio_lengths, audios = format_audio(segm_wav)

    # If no data_dir argument given, assume .csv file follows format of col 1 - audio path & col 2 - transcript
    else:
        transcriptions = []
        task = 3
        # For Task 3 - Sentences (Assumed .csv format)
        for word in word_ts:
            transcriptions.append(word['Text'])
        segm_wav = sw.wav_manip_long(word_ts)
        audio_lengths, audios = format_audio(segm_wav)

    #######################################################################################
    # Model Specific Code
    model = SALM.from_pretrained('nvidia/canary-qwen-2.5b').bfloat16().eval().to(device)
    answer_ids = model.generate(
        prompts=[
            [{"role": "user", "content": f"Transcribe the following: {model.audio_locator_tag}"}]]*len(audios),
                audios=audios.to(device, non_blocking=True),
                audio_lens = audio_lengths.to(device, non_blocking=True),
                max_new_tokens=128,
    )
    results = [model.tokenizer.ids_to_text(oids) for oids in answer_ids.cpu()]
    # print(results)
    if task == 1:
        sw.workbook_write(word_ts,results, task, script_name)
    else: 
        sw.evaluation(results,transcriptions)
    

if __name__ == '__main__':
    main()
