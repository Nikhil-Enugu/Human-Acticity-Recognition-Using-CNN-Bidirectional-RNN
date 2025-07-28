# Human-Acticity-Recognition-Using-CNN-Bidirectional-RNN

This project implements a deep learning model for action recognition in videos using a combination of a Convolutional Neural Network (CNN) and a Bidirectional Recurrent Neural Network (RNN). The model is trained and evaluated on the **HMDB51 dataset**. 🚀

-----

## Overview

The model first uses a pre-trained **MNASNet** CNN to extract spatial features from individual frames of a video. These features are then fed into a **Bidirectional RNN**, which processes the sequence of frames to capture temporal information. The final output is a classification of the action being performed in the video.

This project is built using **PyTorch** and **PyTorch Lightning** for efficient model creation and training.

-----

## Dataset

This project uses the **HMDB51 dataset**, a collection of videos from various sources, categorized into 51 action classes. The provided notebook includes code to automatically download and extract the dataset.

-----

## Requirements

To run this project, you'll need the following libraries:

  * Python 3
  * PyTorch
  * PyTorch Lightning
  * torchmetrics
  * torchvision
  * moviepy
  * av
  * imageio
  * NumPy
  * Matplotlib
  * OpenCV

You can install the necessary packages using pip:

```bash
pip install lightning torchmetrics moviepy av imageio numpy matplotlib opencv-python
```

-----

## Usage

1.  **Clone the repository**:

    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Run the Jupyter Notebook**:
    The main code is in the `Copy of cnn+rnn.ipynb` notebook. Open and run the notebook in a Jupyter environment. The notebook will handle all the steps, from downloading the data to training and evaluating the model.

-----

## Model Architecture 🤖

The `CNNBidirectionalRNN` model is composed of:

1.  **CNN Feature Extractor**: A pre-trained **MNASNet 0.5** model, with its classifier layer replaced by a linear layer to produce a 256-dimensional embedding for each frame.
2.  **Bidirectional RNN**: Processes the sequence of frame embeddings in both forward and backward directions to capture temporal context.
3.  **Transition Layer**: A 1D Convolutional layer combines the outputs from the RNNs.
4.  **Classifier**: A final fully connected layer classifies the video into one of the 51 action classes.

-----

## Results 📊

The model was trained for **64 epochs** and achieved the following performance on the test set:

  * **Test Accuracy**: \~92.6%
  * **Test Loss**: \~0.70

### Training and Validation Curves

The following plots show the training and validation loss and accuracy over the training epochs:

**Loss Curves**

**Accuracy Curves**

-----

## Example Output

Here is an example of the model's prediction on a sample video from the test set. The GIF shows the predicted action and the ground truth label.
