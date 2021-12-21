from pydub import AudioSegment
import os


def match_target_amplitude(sound, target_dBFS):
    """
    Standardize volume of audio clip.
    Taken from here: https://github.com/jiaaro/pydub/issues/90
    """
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def load_raw_audio():
    """
    Load raw audio files for speech synthesis.
    Taken from here: https://www.deeplearning.ai/dp-generate-audio-dataset-738Z#16
    """
    activates = []
    backgrounds = []
    negatives = []

    for filename in os.listdir("./data/raw_data/activates"):
        if filename.endswith("wav"):
            activate = AudioSegment.from_wav("./data/raw_data/activates/"+filename)
            activates.append(activate)

    for filename in os.listdir("./data/raw_data/backgrounds"):
        if filename.endswith("wav"):
            background = AudioSegment.from_wav("./data/raw_data/backgrounds/"+filename)
            backgrounds.append(background)

    for filename in os.listdir("./data/raw_data/negatives"):
        if filename.endswith("wav"):
            negative = AudioSegment.from_wav("./data/raw_data/negatives/"+filename)
            negatives.append(negative)

    return activates, negatives, backgrounds