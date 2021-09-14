import tensorflow as tf
import os
import matplotlib.pyplot as plt
from train_manual import preprocess_dataset, techniques
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # use GPU instead of AVX
model = tf.keras.models.load_model("savedModel")


if __name__ == "__main__":
  model.summary()


  dir_path = os.path.dirname(os.path.realpath(__file__))
  path = os.path.join(dir_path, "samples/manual")

  sample_file = "samples/manual/tasto/tasto37.wav"
  cur_time = time.time()
  sample_ds = preprocess_dataset([sample_file])

  for spectrogram, label in sample_ds.batch(1):
    prediction = model(spectrogram)
    print(f"Time required = {time.time() - cur_time} seconds")
    plt.bar(techniques, tf.nn.softmax(prediction[0]))
    plt.title(f'Predictions for "{techniques[label[0]]}"')
    plt.show()