from matplotlib import pyplot as plt
from src.audio_utils import *
from src.plots import *
import numpy as np


DING_FILE = "audio_examples/chime.wav"


def predict_triggerword(model, filename):
    """
    Use the model to state the probabilites of "activate" word occudence
    in the current audio clip.

    :param model: (Keras.model) - pretrained CNN model
    :param filename: (str) - path to the audio clip

    :return: (np.ndarray) - array of predictions for each timestep
    """
    # Plot the spectrogram.
    plt.subplot(2, 1, 1)
    plt.rcParams["figure.figsize"] = (12,5)
    plt.style.use("default")
    x = get_spectrogram(filename)

    # Make predictions.
    x  = x.swapaxes(0,1) # the model takes those parameters reversed
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    
    # Plot the probability graph.
    plt.style.use("seaborn")
    plt.subplot(2, 1, 2)
    plt.plot(predictions[0,:,0])
    plt.title("Probability of 'Activate' Word Occurence")
    plt.ylabel("Probability")
    plt.xlabel("Timesteps")

    plt.tight_layout()
    plt.show()
    return predictions


def ding_on_activate(filename, predictions, threshold):
    """
    Add "ding" sound after the predicted activate sound.
    This sound is overlayed.
    Save the output audio clip with the name: 

    :param filename: (str) - path to the audio clip
    :param predictions: (np.ndarray) - array of predictions for each timestep
    :param threshold: (double) - if probability is greater than threshold, then consider the word to be "activate"
    """
    audio_clip = AudioSegment.from_wav(filename)
    ding = AudioSegment.from_wav(DING_FILE)
    Ty = predictions.shape[1]

    # Initialize the number of consecutive output steps to 0
    consecutive_timesteps = 0

    # Loop over the output steps in the y
    for i in range(Ty):
        # Increment consecutive output steps
        consecutive_timesteps += 1
        # If prediction is higher than the threshold and more than 75 consecutive output steps have passed
        if predictions[0,i,0] > threshold and consecutive_timesteps > 75:
            # Superpose audio and background using pydub
            audio_clip = audio_clip.overlay(ding, position = ((i / Ty) * audio_clip.duration_seconds) * 1000)
            # Reset consecutive output steps to 0
            consecutive_timesteps = 0
        
    audio_clip.export("./outputs/ding_output.wav", format='wav')