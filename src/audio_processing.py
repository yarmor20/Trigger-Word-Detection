from src.audio_utils import *
from src.plots import *
import numpy as np


Ty = 1375 # The number of time steps in the output of our model (produced by GRU)


def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.
    
    :param segment_ms: (int) - the duration of the audio clip in ms ("ms" stands for "milliseconds")
    :return: segment_time - (tuple) - a tuple of (segment_start, segment_end) in ms
    """
    
    segment_start = np.random.randint(low=0, high=10000 - segment_ms)   # Make sure segment doesn't run past the 10sec background 
    segment_end = segment_start + segment_ms - 1
    
    return (segment_start, segment_end)


def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.
    
    :param segment_time: (np.array) - a tuple of (segment_start, segment_end) for the new segment
    :param previous_segments: (np.ndarray) - a list of tuples of (segment_start, segment_end) for the existing segments
    :return: (bool) - True if the time segment overlaps with any of the existing segments, False otherwise
    """
    
    segment_start, segment_end = segment_time
    
    # Initialize overlap as a "False" flag.
    overlap = False
    
    # Loop over the previous_segments start and end times.
    # Compare start/end times and set the flag to True if there is an overlap
    for previous_start, previous_end in previous_segments:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True

    return overlap


def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the 
    audio segment does not overlap with existing segments.
    
    :param background: (np.array) - a 10 second background audio recording.  
    :param audio_clip: (np.array) - the audio clip to be inserted/overlaid. 
    :param previous_segments: (np.array) - times where audio segments have already been placed
    
    :return: new_background - (np.array) - the updated background audio
    """
    
    # Get the duration of the audio clip in ms
    segment_ms = len(audio_clip)
    
    # Use one of the helper functions to pick a random time segment onto which to insert 
    # the new audio clip.
    segment_time = get_random_time_segment(segment_ms)
    
    # Check if the new segment_time overlaps with one of the previous_segments. If so, keep 
    # picking new segment_time at random until it doesn't overlap.
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)

    # Add the new segment_time to the list of previous_segments
    previous_segments.append(segment_time)
    
    # Superpose audio segment and background
    new_background = background.overlay(audio_clip, position = segment_time[0])
    
    return new_background, segment_time


def insert_ones(y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment 
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 following labels should be ones.
    
    :param y: (np.array) - numpy array of shape (1, Ty), the labels of the training example (0s as an input)
    :param segment_end_ms: (int) - the end time of the segment in ms
    :return: y - (np.array) - updated labels
    """
    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    
    # Add 1 to the correct index in the background label (y)
    for i in range(segment_end_y + 1, segment_end_y + 51):
        if i < Ty:
            y[0, i] = 1
    
    return y


def create_training_example(background, activates, negatives):
    """
    Creates a training example with a given background, activates, and negatives.
    
    :param background: (np.array) - a 10 second background audio recording
    :param activates: (np.ndarray) - a list of audio segments of the word "activate"
    :param negatives: (np.ndarray) - a list of audio segments of random words that are not "activate"
    
    :return: x - (np.ndarray) - the spectrogram of the training example
    :return: y - (np.ndarray) - the label at each time step of the spectrogram
    """
    # Set the random seed
    np.random.seed(18)
    
    # Make background quieter
    background = background - 20

    # Initialize y (label vector) of zeros
    y = np.zeros((1, Ty))

    # Initialize segment times as empty list
    previous_segments = []
    
    # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]
    
    # Loop over randomly selected "activate" clips and insert in background
    for random_activate in random_activates:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        _, segment_end = segment_time
        # Insert labels in "y"
        y = insert_ones(y, segment_end_ms=segment_end)

    # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    # Step 4: Loop over randomly selected negative clips and insert in background
    for random_negative in random_negatives:
        # Insert the audio clip on the background 
        background, _ = insert_audio_clip(background, random_negative, previous_segments)
    
    # Standardize the volume of the audio clip 
    background = match_target_amplitude(background, -20.0)

    # Export new training example 
    _ = background.export("./outputs/train.wav", format="wav")
    print("File (train.wav) was saved in your directory.")

    # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
    x = get_spectrogram("./outputs/train.wav")
    
    return x, y


if __name__ == "__main__":
    # Load the audio clips.
    activates, negatives, backgrounds = load_raw_audio()

    # Generate new dataset.
    for i, background in enumerate(backgrounds):
        X, Y = create_training_example(background, activates, negatives)
        np.save(f"./data/XY_test/X{i}", np.array(X))
        np.save(f"./data/XY_test/Y{i}", np.array(Y))