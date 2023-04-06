# streaming-stt


This is a simple streaming STT service using the Google Cloud Speech API.

## Setup

You'll need to set up a Google Cloud project and enable the Speech API. Then, you'll need to create a service account and download the JSON key file. Finally, you'll need to set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of the JSON key file. 

Follow the instructions here (https://cloud.google.com/speech-to-text/docs/before-you-begin) to set up your project and service account.

From this directory, you can store the credentials in a file called creds.json and run the following command:

```bash
source ./env.sh
```

## Install the package

Within your python environment of choice, run: 
```bash
pip install -e .
```

## Usage

To run the streaming STT service and stream the output to a file, run:
```bash
python stt.py speech.txt --keywords apple banana
```

New instances of the keywords will be written to `keywords.txt` as soon as they're found in the text. 

To view the streaming STT service output in real time, run:
```bash
watch -n 0.1 tail -n 10 speech.txt
```

### Sound Classification

To run sound classification with yamnet and stream the output to a file, run:
```bash
python yamnet.py --output_file sound.txt
```
