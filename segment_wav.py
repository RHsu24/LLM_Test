# not designed to be run as a standalone file
# designed to be run using minimal extra packages
import csv
import wave
import math
import io
import re
import os
import evaluate
import sys


def clean_transcript(transcript):
    # Clean up manual annotator symbols
    clean_ts = re.sub(r'\[O\]|\[\]|XXX|\[X\]|\[.\]|\[..\]','', transcript)
    return clean_ts

def normalise_word(word):
    cleaned_text = re.sub(r"[!?.,'_\-]+", " ", word)  # Remove extra punctuation
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()  # Remove extra spaces
    return cleaned_text

def evaluation(prediction,reference): 
    wer = evaluate.load("wer")
    frame = None
    try:
        caller_file = sys._getframe(1).f_code.co_filename
        canary_chk = os.path.basename(caller_file)
    finally:
        del caller_file

    if canary_chk != 'canary_qwen2_5b.py':
        character = evaluate.load("character")
    predict_clean = []
    ref_clean = []
    predict_count =  ref_count = 0
    empty_ref = 0
    for i, (pred, ref) in enumerate(zip(prediction,reference)):
        cleaned_pred = normalise_word(pred)
        cleaned_ref = clean_transcript(ref)
        cleaned_ref = normalise_word(cleaned_ref)
    # Manually check for None, whitespace or empty string to catch before computing
    # This includes strings empty as a result of removed transcribed annotations
        if (not cleaned_ref  or not cleaned_ref.strip() ):
            empty_ref += 1
            continue
        predict_clean.append(cleaned_pred)
        ref_clean.append(cleaned_ref)
        predict_count += len(cleaned_pred.split())
        ref_count += len(cleaned_ref.split())
    
    if canary_chk != 'canary_qwen2_5b.py':
        char_results = character.compute(references=ref_clean,predictions=predict_clean)
        print(f'Character Error Stats (using Levenshtein distance):') 
        for key, value in char_results.items():
            print(f"{key}: {value}")
    wer_score = wer.compute(references=ref_clean,predictions=predict_clean)
    # Currently, output to command terminal
    print(f'Word Error Rate: {wer_score}/word over span of {ref_count} words.')
    print(f'There were {empty_ref} empty reference strings omitted from calculation.')


def workbook_write(word_files, results, task, name):
    script_name = os.path.splitext(name)[0]
    wb_name = str(script_name) + '_test.csv'
    fieldnames = ['Transcriptions', 'Predictions', 'Binary_eval']
    csv_rows = []
    for i, (word,r) in enumerate(zip(word_files,results)):
        if task == 1:
            read_word = word['text']
        elif task == 3:
            read_word = word
        #clean up the words for comparison
        word_clean = normalise_word(read_word)      
        word_cmp = normalise_word(clean_transcript(read_word))
        asr_clean = normalise_word(r)
  
        rows = [word_clean, asr_clean, word_cmp.casefold() == asr_clean.casefold()]
        csv_rows.append(rows)

    with open(wb_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(csv_rows)


def wav_manip(wav, dict): # manipulate short, individual audio files from one long audio file
    word_count = 1
    word_buffer = []
    print(f"Processing Audio Files into Memory...")
    try: 
        #print(f"Checking file: {wav}, Size: {os.path.getsize(wav)} bytes")
        with wave.open(wav,'rb') as wav_file:
            channels = wav_file.getnchannels()  # Mono or Stereo
            sample_width = wav_file.getsampwidth()  # Bytes
            frame_rate = wav_file.getframerate()    # Sampling Frequency
            comp_type = wav_file.getcomptype()
            comp_name = wav_file.getcompname()

            for row in dict:
                buffer = io.BytesIO()
                t_start =   float(row['tmin'])
                t_end = float(row['tmax'])
                word_count += 1
                start_frame = math.floor(t_start*frame_rate)
                end_frame = math.ceil(t_end*frame_rate)
                word_frames_tot = end_frame - start_frame
                wav_file.setpos(start_frame)
                word_bytes = wav_file.readframes(word_frames_tot)

                with wave.open(buffer,'wb') as wav_word:
                    wav_word.setparams((channels, sample_width, frame_rate, word_frames_tot,comp_type,comp_name))
                    wav_word.writeframes(word_bytes)
                    word_buffer.append(buffer)
    except FileNotFoundError:
        print(f"Error: The file {wav_file} was not found.")
    except wave.Error as e:
        # Catches invalid formats, corrupt files, or non-WAV files
        print(f"Error: Could not read WAV file. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    print(f"Parsed Audio into Memory")
    return word_buffer

def wav_manip_long(dict_list): # manipulate a long list (listed in .csv) of audio files
    word_buffer = []
    print(f"Processing Audio Files into Memory...")
    for row in dict_list:
        wav = row['Audio']
        buffer = io.BytesIO()
        t_start = float(row['tmin'])
        t_end = float(row['tmax'])
        
        try: 
            #print(f"Checking file: {wav}, Size: {os.path.getsize(wav)} bytes")
            with wave.open(wav,'rb') as wav_file:
                # .wav file data
                channels = wav_file.getnchannels()  # Mono or Stereo
                sample_width = wav_file.getsampwidth()  # Bytes
                frame_rate = wav_file.getframerate()    # Sampling Frequency
                comp_type = wav_file.getcomptype()
                comp_name = wav_file.getcompname()
                
                # go to actual transcribed audio
                start_frame = math.floor(t_start*frame_rate)
                end_frame = math.ceil(t_end*frame_rate)
                word_frames_tot = end_frame - start_frame
                wav_file.setpos(start_frame)
                word_bytes = wav_file.readframes(word_frames_tot)
                # Save to buffer
                with wave.open(buffer,'wb') as wav_word:
                    wav_word.setparams((channels, sample_width, frame_rate, word_frames_tot,comp_type,comp_name))
                    wav_word.writeframes(word_bytes)
                    word_buffer.append(buffer)

        except FileNotFoundError:
            print(f"Error: The file {wav_file} was not found.")
        except wave.Error as e:
            # Catches invalid formats, corrupt files, or non-WAV files
            print(f"Error: Could not read WAV file. {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    print(f"Parsed Audio into Memory")
    return word_buffer

def read_transcript(word_ts):
# Audio Transcript -> Full patient audio file, Word Transcript -> CSV file with timestamps of words
    ts_data = []
    try:
        with open(word_ts, mode='r',newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            ts_data = list(reader)
    except FileNotFoundError:
        print(f"Error: The file {word_ts} was not found.")
    except PermissionError:
        # Catches invalid formats, corrupt files, or non-WAV files
        print(f"Error: Insufficient permission to read CSV file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return ts_data
