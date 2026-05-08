# not designed to be run as a standalone file
# designed to be run using minimal extra packages
import csv
import wave
import math
import io
import re
import os
import xlsxwriter
import evaluate

def normalise_word(word):
    cleaned_text = re.sub(r"[!?.,'_\-]+", " ", word)  # Remove extra punctuation
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()  # Remove extra spaces
    return cleaned_text

def evaluation(prediction,reference):
    wer = evaluate.load("wer")
    character = evaluate.load("character")
    predict_clean = []
    ref_clean = []
    for i, (pred, ref) in enumerate(zip(prediction,reference)):
        predict_clean.append(normalise_word(pred))
        ref_clean.append(normalise_word(ref))
        predict_count += len(predict_clean[i].split())
        ref_count += len(ref_clean[i].split())

    char_results = character.compute(references=ref_clean,predictions=ref_clean)
    wer_score = wer.compute(references=ref_clean,predictions=predict_clean)
    # Currently, output to command terminal
    print(f'Word Error Rate: {wer_score}/word over span of {ref_count} words.')
    print(f'Character Error Stats (using Levenshtein distance): {char_results}')

def workbook_write(word_files, results, task, name):
    script_name = os.path.splitext(name)[0]
    wb_name = str(script_name) + '_test.xlsx'
    workbook = xlsxwriter.Workbook(wb_name)
    worksheet = workbook.add_worksheet()
    bold = workbook.add_format({'bold':True})
    worksheet.write('A1','Expected', bold)
    worksheet.write('B1','Recognised', bold)
    worksheet.write('C1', "Binary_eval", bold)
    if task == 1:
        for i, (word,r) in enumerate(zip(word_files,results)):
            read_word = word['text']
            #clean up the words for comparison - combine multi-words into one word 
            word_clean = normalise_word(read_word)
            asr_clean = normalise_word(r)
            worksheet.write(i+1,0,word_clean)
            worksheet.write(i+1,1,asr_clean)
            worksheet.write(i+1,2, word_clean.casefold() == asr_clean.casefold())
    elif task == 3:
        placeholder = []

    workbook.close()

def wav_manip(wav, dict):
    word_count = 1
    word_buffer = []
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
                txt_count =  f"{word_count:03d}"
                wav_name = row['text']
                t_start =   float(row['tmin'])
                t_end = float(row['tmax'])
                # print(f'{new_wav_name}, {t_start}, {t_end}')
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
