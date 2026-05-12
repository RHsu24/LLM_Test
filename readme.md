### README for csv_parse.py

```csv_parse.py``` is designed to read in a directory containing any number of transcription .csv files with timestamps, and their associated audio .wav files. It will then compile a complete list of every valid set of transcription files and associate audio .wav files into a 'master' transcription file.



Before running:

* Ensure that all the column headers in every .csv file are standardised. If you need to adjust the header names, see ==Line 24== (for reading base transcription files) and adjust header names accordingly.
* This script is designed for downstream use in testing ASR models, and as such keywords (such as the master-transcription file headers) are important. Please think carefully about renaming them.
* Ensure all your .csv files have a **single** corresponding .wav file. It will throw an error if it cannot find a matching .wav file, but **will not** if it can find more than 1 matching .wav file.
* Run this .py script on the same OS as downstream testing scripts. This is because we are saving pathnames into a csv and delimiting between '\\' (Windows) and '/' (MacOS/Linux) may become an issue.



How To Run:

```python csv_parse.py TRANSCRIPT_DIR --file_suffix FILE_SUFFIX```

```TRANSCRIPT_DIR``` -> Required 

* Directory containing any number of sets of transcription .csv files and their associated .wav file

```--file_suffix FILE_SUFFIX``` -> Optional Argument

* &#x20;This is any text in the filename that comes after the set name (to match to .wav file) but before the file format (.csv). For example, if my transcription file is 228_task3_child.csv and my audio file is 228_task3.wav, then the FILE_SUFFIX is ```_child```. If no optional argument input, please ensure file names (excluding file format) are **EXACTLY** the same, and no extra .csv files are in the ```TRANSCRIPT_DIR``` directory.

