import pandas as pd
import argparse
import glob
import sys
import os
from pathlib import Path

def search_for_wav(transcript, search_name, root_dir):
    key_words = transcript.rsplit(search_name,1)
    wav_name = key_words[0] + '.wav'
    wav = glob.glob(wav_name,root_dir=root_dir)
    if wav is not None:
        path = (Path(root_dir) / wav[0]).resolve()
        #print(f'Found .wav file path: {path}')
        return path
    else:
        sys.exit(f"Error: Could not find matching .wav file to transcription {transcript}."
                "Ensure all transcriptions have valid and existing .wav file")
        
def parse_transcript(word_ts):
# Transcript -> CSV file with timestamps of sentences
# Parse through CSV and return only necessary data
    try:
        df = pd.read_csv(word_ts, usecols=['transcription_hu', 'start_hu', 'end_hu'], 
                             on_bad_lines='error')
        df.columns = ['Text', 't_start', 't_end']
    except FileNotFoundError:
        print(f"Error: The file {word_ts} was not found.")
    except PermissionError:
        # Catches invalid formats, corrupt files, or non-WAV files
        print(f"Error: Insufficient permission to read CSV file.")
    except pd.errors.ParserError:
        print(f"Error: There was a parsing error (e.g., malformed data).")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return df

def write_to_csv(transcriptions, audio_files):
    # It will need to write to a .csv file with col 1 containing 
    # audio PATH, col 2 containing the transcription. and col 3/4 with 
    # t_start and t_end
    # fieldnames = ['Audio', 'Text','t_start','t_end']
    pd_list = []
    for i, (ts, audio) in enumerate(zip(transcriptions, audio_files)):
        df = parse_transcript(ts)
        df.insert(loc=0, column='Audio', value=audio)
        pd_list.append(df)
    pd_master = pd.concat(pd_list, ignore_index=True)
    save_file_name = 'transcript_master.csv'
    pd_master.to_csv(save_file_name,encoding='utf-8', index=False, header=True)
    print(f'Saved master transcription file: {save_file_name} to current directory')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('transcript_dir', type=str,
                        help="Directory to the list of .csv transcription files and audio")
    parser.add_argument('--file_suffix', type=str, nargs='?',
                        help="Suffix of .csv file to aid in searching")
    
    args = parser.parse_args()
    list_dir = args.transcript_dir
    suffix = ''
    if (args.file_suffix) is not None: suffix = args.file_suffix 
    if not(os.path.exists(args.transcript_dir)):
        sys.exit("Error: Could not find files on path. Ensure file path is valid")    

    audio_files = []
    transcriptions = []
    search_name = suffix + '.csv'

    # Ensure paths are saved as ABSOLUTE paths for other scripts to access
    for files in glob.glob("*.csv",root_dir=list_dir):
        if files.endswith(search_name):
            path = (Path(list_dir) / files).resolve()
            transcriptions.append(path)
            wav_name = search_for_wav(files, search_name, list_dir)
            audio_files.append(wav_name)
    
    print(f'Found {len(audio_files)} audio files and {len(transcriptions)} transcription files\n' 
          'Now writing to csv...')
    write_to_csv(transcriptions, audio_files)


if __name__ == '__main__':
    main()
