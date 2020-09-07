import librosa
import numpy as np
import pickle

# Authors Ben Davies, Rory Hicks, Vicky Norman. University of Bristol. Jan 2019.


def melspectrogram(audio):
    spec = librosa.stft(audio, n_fft=512, window='hann', hop_length=256, win_length=512, pad_mode='constant')
    mel_basis = librosa.filters.mel(sr=22050, n_fft=512, n_mels=80)
    mel_spec = np.dot(mel_basis, np.abs(spec))
    return np.log(mel_spec + 1e-6)

# Function to create pickle datasets with the data and labels converted offline in order to speed computation
def create_spec_dataset():


    # Load train and test data
    with open('music_genres_dataset.pkl', 'rb') as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)

    # Function to convert data into 80x80 spectrograms and labels into 1x10 arrays
    def convertData(dataset):

        # Apply spectrogram function to each audio segment in train set
        data = np.array(dataset['data'])
        specs = np.zeros((np.shape(data)[0], 80, 80))
        for i in range(np.shape(data)[0]):
            specs[i, :, :] = melspectrogram(data[i])

        # Update the data key of train dictionary
        dataset['data'] = specs

        # Convert 0-based integer label to 1x10 array with index of non-zero value representing class
        labels = dataset['labels']
        newLabels = np.zeros((len(labels), 10))
        for k in range(len(labels)):
            newLabels[k, labels[k]] = 1

        # Update labels key of train dictionary
        dataset['labels'] = newLabels

        return dataset


    # Convert train and test sets
    train_set = convertData(train_set)
    test_set = convertData(test_set)

    # Create new pickle datasets for training and testing
    pickle.dump(train_set, open("train_spectogram_dataset.pkl", "wb"), protocol=2)
    pickle.dump(test_set, open("test_spectogram_dataset.pkl", "wb"), protocol=2)

# Function to create pickle datasets of the train set with further augmented samples added
def create_augmented_spec_dataset():

    # Load train set
    with open('music_genres_dataset.pkl', 'rb') as f:
        train_set = pickle.load(f)

    # Function to augment audio segment with a pitch shift of 'n_steps' semitones and a time stretch of factor 'rate'
    def augment_audio(audio, rate, n_steps):

        # Augmentation
        audio1 = librosa.effects.pitch_shift(audio, 22050, n_steps, bins_per_octave=12)
        audio2 = librosa.effects.time_stretch(audio1, rate)

        # Adjusting array sizes to match original segment length.
        if len(audio2) > len(audio):
            audio3 = audio2[:len(audio)]
        elif len(audio2) < len(audio):
            extra = np.zeros(len(audio)-len(audio2))
            audio3 = np.concatenate((audio2, extra))
        else:
            audio3 = audio2

        return audio3

    # Load data
    data = np.array(train_set['data'])
    labels = np.array(train_set['labels'])
    track_id = np.array(train_set['track_id'])
    # sz = np.shape(data)[1]*16

    # Initialise arrays with correct sizes to fill with augmented segments
    augmented_audio_segs = np.zeros((36000, np.shape(data)[1]))
    augmented_audio_labels = np.zeros(36000)
    augmented_audio_IDs = np.zeros(36000)

    # Initialise parameters for augmentation
    specs = np.zeros((47250, 80, 80))
    rates = [0.2, 0.5, 1.2, 1.5]
    n_steps = [-5, -2, 2, 5]

    # For each track 3 segments are taken and augmented with every combination of augmentation parameters to create an
    # extra 48 samples per track.
    z = 0
    for i in range(11250):
        if i % 5 == 0:
            print(i)
            for k in range(len(rates)):
                for n in range(len(n_steps)):
                    aug_audio = augment_audio(data[i], rates[k], n_steps[n])
                    augmented_audio_segs[z] = aug_audio
                    augmented_audio_labels[z] = labels[i]
                    augmented_audio_IDs[z] = track_id[i]
                    z = z + 1

    # Amalgamate newly created samples with original audio segments
    data = np.vstack((data, augmented_audio_segs))
    print(np.shape(data), np.shape(labels), np.shape(track_id))
    labels = np.hstack((labels, augmented_audio_labels))
    track_id = np.hstack((track_id, augmented_audio_IDs))
    print(np.shape(data), np.shape(labels), np.shape(track_id))

    # Convert data to 80x80 spectrogram images
    for i in range(47250):
        specs[i, :, :] = melspectrogram(data[i])


    # Update train set
    train_set['data'] = specs
    train_set['labels'] = labels
    train_set['track_id'] = track_id
    print(train_set)

    # Had to split up augmented set as it was too large to singularly pickle
    train1 = {}
    train1['data'] = specs[23625:]
    train1['labels'] = labels[23625:]
    train1['track_id'] = track_id[23625:]

    train2 = {}
    train2['data'] = specs[:-23625]
    train2['labels'] = labels[:-23625]
    train2['track_id'] = track_id[:-23625]

    # Convert labels into discrete 1x10 arrays with the the non-zero index representing the class
    labels1 = train1['labels']
    newLabels1 = np.zeros((len(labels1), 10))
    for k in range(len(labels1)):
        newLabels1[k, labels1[k]] = 1
    train1['labels'] = newLabels1
    labels2 = train2['labels']
    newLabels2 = np.zeros((len(labels2), 10))
    for k in range(len(labels2)):
        newLabels2[k, labels2[k]] = 1
    train2['labels'] = newLabels2

    # Create two new pickle databases to store augmented training set
    pickle.dump(train1, open("augment_train_spectogram_dataset1.pkl", "wb"), protocol=2)
    pickle.dump(train2, open("augment_train_spectogram_dataset2.pkl", "wb"), protocol=2)
