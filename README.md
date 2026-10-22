# Cats and Dogs Classifier

A computer vision project I built for my Artificial Intelligence Systems course during Erasmus. It compares three neural network architectures for classifying cats and dogs, then brings a convolutional model into the browser for predictions from a webcam.

The repository includes the training notebook, a Python export and a browser demo with model weights. You can open the demo without training a model first.

## Run the webcam demo

Clone the repository and start a local server from its root:

```bash
git clone https://github.com/nachopalomares/cat-dog-classifier.git
cd cat-dog-classifier
python3 -m http.server 8000 --bind 127.0.0.1 --directory web
```

Open **http://localhost:8000**, allow camera access and wait for the model to load. Point the camera at a cat, dog or a photo of one. Use **Change camera** to switch between the front and rear cameras on supported devices.

The browser resizes each frame to 100 × 100 pixels, converts it to grayscale and runs inference with TensorFlow.js. A sigmoid output at or below `0.5` is displayed as **Cat**; a higher value is displayed as **Dog**.

Predictions run locally in the browser; the demo does not send camera frames to a server. An internet connection is needed to load TensorFlow.js and Bootstrap from their CDNs. Camera access requires localhost or HTTPS when hosting the demo elsewhere.

## Training workflow

The notebook uses the `cats_vs_dogs` dataset from TensorFlow Datasets. Images are resized to **100 × 100**, converted to a single grayscale channel and normalized to the **0–1** range.

Three architectures are compared, each with and without data augmentation:

| Model | Architecture |
| --- | --- |
| Dense | Flattened image followed by two 150-unit ReLU layers and a sigmoid output. |
| CNN | Three convolutional blocks with 32, 64 and 128 filters, max pooling, a 100-unit dense layer and a sigmoid output. |
| CNN with dropout | The same convolutional blocks, 50% dropout and a 250-unit dense layer before the output. |

Training uses Adam, binary cross-entropy, batches of 32 and 100 epochs. Augmentation adds rotation, shifts, shear, zoom and horizontal and vertical flips. TensorBoard records accuracy and loss for each experiment.

The baseline experiments use a 15% validation split. The augmented experiments use the first 19,700 samples for training and the remaining samples for validation.

## Results from the original run

These are the final-epoch validation accuracies recorded in the original notebook output:

| Model | Without augmentation | With augmentation |
| --- | ---: | ---: |
| Dense | 59.14% | 49.21% |
| CNN | 81.98% | 84.92% |
| CNN with dropout | 83.95% | 82.43% |

The CNNs performed better than the dense baseline. The augmented CNN reached the highest final validation accuracy in this run. The two training setups use slightly different validation splits, so the table describes these experiments rather than a controlled benchmark. Webcam predictions also depend on lighting, framing and background.

## Explore the notebook

Open [dogsandcats.ipynb](dogsandcats.ipynb) in Google Colab. The project was developed there using a GPU runtime. Its Python dependencies are:

```python
%pip install tensorflow tensorflow-datasets tensorflowjs opencv-python matplotlib numpy
```

Run the cells in order to load the dataset, prepare the images and train the six model variants. Dataset images are downloaded by TensorFlow Datasets and are not included in this repository. The notebook keeps the preprocessed dataset in memory, so a GPU runtime with sufficient RAM is useful.

The final cells save `CNNmodel_AD` as `dogs-cats-cnn-ad.h5` and convert it to TensorFlow.js format in `carpeta_salida/`. To use a new export in the demo, copy its `model.json` and all referenced weight shards into `web/` together. The included browser model accepts inputs with shape `(100, 100, 1)`.

[dogsandcats.py](dogsandcats.py) is the original Colab export for reading the training code outside the notebook. It retains notebook shell commands, so the notebook is the main entry point for training.

## Repository structure

```text
dogsandcats.ipynb       Training notebook, organized into sections
dogsandcats.py          Python export of the original notebook
web/
  index.html           Webcam demo and inference logic
  model.json           TensorFlow.js model definition
  group1-shard*.bin     Exported model weights
  pet.png              Demo icon
  favicon*             Browser icons
```

The notebook's saved outputs have been cleared to keep it easy to browse. The results above come from its original training logs.

## Tools

Python, TensorFlow/Keras, TensorFlow Datasets, OpenCV, NumPy, Matplotlib and TensorBoard for training; JavaScript, TensorFlow.js and Bootstrap for the browser demo.
