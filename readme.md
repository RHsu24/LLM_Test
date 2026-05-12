### README for csv\_parse.py

==csv\_parse.py== is designed to read in a directory containing any number of transcription .csv files with timestamps, and their associated audio .wav files. It will then compile a complete list of every valid set of transcription files and associate audio .wav files into a 'master' transcription file.



Before running:

* Ensure that all the column headers in every .csv file are standardised. If you need to adjust the header names, see ==Line 24== (for reading base transcription files)
* This script is designed for downstream use in testing ASR models, and as such keywords (such as the master-transcription file headers) are important. Please think carefully about renaming them.
* Ensure all your .csv files have a **single** corresponding .wav file. It will throw an error if it cannot find a matching .wav file, but **will not** if it can find more than 1 matching .wav file.
* Run this .py script on the same OS as downstream testing scripts. This is because we are saving pathnames into a csv and delimiting between '\\' (Windows) and '/' (MacOS/Linux) may become an issue.



How To Run:

```python csv\_parse.py {TRANSCRIPT\_DIR} --file\_suffix {FILE\_SUFFIX}```

```{TRANSCRIPT\_DIR}``` -> Required 

* Directory containing any number of sets of transcription .csv files and their associated .wav file

```--file\_suffix file\_suffix``` -> Optional Argument

* &#x20;This is any text in the filename that comes after the set name (to match to .wav file) but before the file format (.csv). For example, if my transcription file is 228\_task3\_child.csv and my audio file is 228\_task3.wav, then the file\_suffix is ```\_child```

