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

Output:
The output will be ```transcript_master.csv``` by default, containing columns \[Audio, Text, tmin, tmax]\

### README FOR LLM_TEST REPOSITORY

1. ##### Introduction



This repository was used to test some of the top ASR (Automatic Speed Recognition) Models and their effectiveness in inferring words and sentences from recorded audio, compared to manual transcription. These scripts have been designed to run on a HPC (High Performance Computing) Cluster, specifically the Katana HPC. These scripts perform batch inference in non-real-time and are focused on tracking accuracy rather than speed.



The current models, with corresponding scripts available to test are:

[Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) with ```Qwen3.py```

[CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) with ```coherelabs_transcribe.py```

[Lexius/Phi-4-multimodal-instruct](https://huggingface.co/Lexius/Phi-4-multimodal-instruct) - with ```Phi_4.py```

[nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) with ```parakeetTDTv2.py```

[nvidia/canary-qwen-2.5b](https://huggingface.co/nvidia/canary-qwen-2.5b) with ```canary_qwen2_5b.py```

[openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) with ```openai_whisper-large_v3.py```

[ibm-granite/granite-speech-4.1-2b](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) with ```ibm_granite4.py```

[mistralai/Voxtral-Small-24B-2507](https://huggingface.co/mistralai/Voxtral-Small-24B-2507#transformers-%25F0%259F%25A4%2597) with ```mistralai_voxel-small24B.py```





##### 2. Before Running:


Set your ```HF_HOME``` environment variable to a location that has sufficient storage. There will be a significant number of parameters being installed the first time you run each script, with Voxtral-Small-24B-2507 being the largest at 24B parameters. This means there will be \~49GB of tensors downloaded and installed.

Do this by (on Linux): ```export HF_HOME='/usr/directory/of/choice'```

Ensure you have several different PYTHON_PATH environment variables and virtual environments set up. These scripts have inherited different package versions and (in the case of canary-qwen-2.5B) **even different Python versions** from the requirements of the ASR models. Many of these scripts will **not** run on newer versions of packages. For more information on script package/version requirements, refer below.

The CohereLabs ASR model is a gated model. You will need a token from HuggingFace for this specific model, and store it in a ```.env``` file as a variable ```HF_TOKEN = YOURTOKEN```



##### 3. REQUIREMENTS:

###### PYTHON VERSION

**3.10.8**

Runs scripts:

* Qwen3.py
* coherelabs_transcribe.py
* Phi_4.py
* parakeetTDTv2.py
* openai/whisper-large-v3
* ibm_granite4.py
* mistralai_voxel-small24B.py





**3.13.2**

Runs scripts:

* canary_qwen2_5b.py



It is possible that some or all of these scripts could run on different versions of Python, but development and testing of the scripts was completed on the specified Python versions. ```canary_qwen2_5b.py``` was explicitly tested on 3.10.8 and was unable to be run. These are the only available Python versions (3.10+) on the Katana HPC, aside from 3.11.



###### PACKAGES 
While most of the ASR models have different package version requirements, there was some attempt to collate model test scripts into as few sets of package requirements as possible. As of now, there are 5 different sets of packages:
* cohereLabs, openai-whisper and mistralai-Voxel-Small - see ```coherelabs_req```
* parakeet-tdt and IBM-Granite-4 - see ```parakeet_reqs.txt```
* Qwen3
* Canary-Qwen
* Phi-4
