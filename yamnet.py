import csv
from pathlib import Path
import time

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pyaudio

SAMPLE_RATE = 16000
CHUNKSIZE = int(SAMPLE_RATE * 1.8) # fixed chunk size


# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])

  return class_names

def load_yamnet():
    # Load the model.
    model = hub.load('https://tfhub.dev/google/yamnet/1')


    class_map_path = model.class_map_path().numpy()
    class_names = class_names_from_csv(class_map_path)
    return model, class_names



def yamnet_infer(output_file: str = "sounds.txt"):
    start_time = time.time()
    output_file = Path(output_file)
    with open(output_file, "w") as f:

        model, class_names = load_yamnet()

        # initialize portaudio
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=CHUNKSIZE)
        try:
            while True:
                # do this as long as you want fresh samples
                # TODO: put samples in a queue and process them in a separate thread
                # to avoid buffer overflows
                data = stream.read(CHUNKSIZE, exception_on_overflow=False)
                waveform = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                waveform = waveform / np.iinfo(np.int16).max

                # Run the model, check the output.
                scores, embeddings, spectrogram = model(waveform)

                scores = scores.numpy()
                spectrogram = spectrogram.numpy()
                infered_class = class_names[scores.mean(axis=0).argmax()]

                top_classes = np.argsort(scores.mean(axis=0))[::-1]
                top = "\t|\t".join(f"{class_names[i]:<15}" for i in top_classes[:5])
                
                f.write(f"{time.time()-start_time},{top}\n")
                f.flush()
                print(f"{time.time()-start_time:<3.3f},{top}")


        except KeyboardInterrupt:
            # close stream
            stream.stop_stream()
            stream.close()
            p.terminate()


if __name__ == "__main__":
    import argbind

    argbind.bind(yamnet_infer, without_prefix=True)

    args = argbind.parse_args()

    with argbind.scope(args):
        yamnet_infer()