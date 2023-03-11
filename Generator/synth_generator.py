import random
import csv
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import statsmodels.api as sm
from scipy.signal import find_peaks
from pathlib import Path
from scipy.signal import butter, lfilter

# Make sure the directory exists, create it if not
path = Path('audiodata')
path.mkdir(parents=True, exist_ok=True)

param_values = []       # Create a list to store the input parameter values
sr = 44100              # Set the sampling rate
n = 1                   # Number of sound files to be generated

def frequency(note):                                                            # Create the frequency from note
    if note == 'c':
        freq = 440 * (2 ** ((60 - 69) / 12))
    elif note == 'd':
        freq = 440 * (2 ** ((62 - 69) / 12))
    elif note == 'e':
        freq = 440 * (2 ** ((64 - 69) / 12))
    elif note == 'f':
        freq = 440 * (2 ** ((65 - 69) / 12))
    elif note == 'g':
        freq = 440 * (2 ** ((67 - 69) / 12))
    elif note == 'a':
        freq = 440 * (2 ** ((69 - 69) / 12))
    elif note == 'b':
        freq = 440 * (2 ** ((71 - 69) / 12))
    else:
        raise ValueError("Invalid note")
    return freq

def oscillator(osc, freq, duration):                                            # Create the oscillator
    if osc == 'sine':
        osc = np.sin(2 * np.pi * np.arange(sr * duration) * freq / sr)
    elif osc == 'square':
        osc = np.sign(np.sin(2 * np.pi * np.arange(sr * duration) * freq / sr))
    elif osc == 'sawtooth':
        osc = 2 * (np.arange(sr * duration) * freq / sr - np.floor(np.arange(sr * duration) * freq / sr)) - 1
    elif osc == 'triangle':
        osc = 2 * np.abs(2 * (np.arange(sr * duration) * freq / sr - np.floor(np.arange(sr * duration) * freq / sr)) - 1) - 1
    elif osc == 'noise':
        osc = np.random.normal(0, 1, duration * sr)
    else:
        raise ValueError("Invalid oscillator type")
    return osc

def oscillator2(osc2, osc2_phase, freq, duration):                              # Create the oscillator 2
    if osc2 == 'sine':
        osc2 = np.sin(2 * np.pi * np.arange(sr * duration) * freq / sr + osc2_phase)
    elif osc2 == 'square':
        osc2 = np.sign(np.sin(2 * np.pi * np.arange(sr * duration) * freq / sr + osc2_phase))
    elif osc2 == 'sawtooth':
        osc2 = 2 * (np.arange(sr * duration) * freq / sr - np.floor(np.arange(sr * duration) * freq / sr)) - 1
    elif osc2 == 'triangle':
        osc2 = 2 * np.abs(2 * (np.arange(sr * duration) * freq / sr - np.floor(np.arange(sr * duration) * freq / sr)) - 1) - 1
    elif osc2 == 'noise':
        osc2 = np.random.normal(0, 1, duration * sr)
    else:
        raise ValueError("Invalid oscillator 2 type")
    return osc2

def envelope(attack, decay, sustain, release, duration):                        # Create the envelope for the note
    envelope = np.zeros(int(sr * duration))
    envelope[:int(attack * sr)] = np.linspace(0, 1, int(attack * sr))
    envelope[int(attack * sr):(int(attack * sr) + int(decay * sr))] = np.linspace(1, sustain, int(decay * sr))
    envelope[(int(attack * sr) + int(decay * sr)):(int(attack * sr) + int(decay * sr) + int(1 * sr))] = sustain
    envelope[(int(attack * sr) + int(decay * sr) + int(1 * sr)):(int(attack * sr) + int(decay * sr) + int(1 * sr) + int(release * sr))] = np.linspace(sustain, 0, int(release * sr))
    return envelope
    
def lfo(mod, duration):                                                         # Create the LFO
    lfo = np.sin(2 * np.pi * np.arange(sr * duration) * mod / sr)
    return lfo

def lp_filter(sound, cutoff, sr=sr, order=2):                                   # Create the LP-Filter
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, sound)
    return y

def synthesizer(note, osc, mod, attack, decay, sustain, release, cutoff):       # Create the sound
    duration = 4                                                                    # Set the maximum duration of the note
    freq = frequency(note)                                                          # Create the frequency frome note
    sound = oscillator(osc, freq, duration)                                         # Create the oscillator 1
    #osc2 = oscillator2(osc2, osc2_phase, freq, duration)                            # Create the oscillator 2
    #sound = (osc1 * mix) + (osc2 * (1 - mix))                                       # Mix the oscillators and apply the amplitudes
    sound *= envelope(attack, decay, sustain, release, duration)                    # Apply the envelope to the oscillators
    sound *= (1 + lfo(mod, duration)) / 2                                           # Apply amplitude modulation
    sound = lp_filter(sound, cutoff, order=4)                                       # Low-Pass Filter
    sound = sound / np.max(np.abs(sound))                                           # Normalize the sound

    pitch = pitchdetection(sound)                                                   # Detect Pitch
    print(pitch)
    return sound

def pitchdetection(sound):
    #N = np.trim_zeros(sound).size                                                   # Signal length in samples
    #sound = sound[0:N]                                                              # Trim Sound to length
    auto = sm.tsa.acf(sound, nlags=2000)                                            # Compute the autocorrelation
    peaks = find_peaks(auto)[0]                                                     # Find peaks of the autocorrelation
    lag = peaks[0]                                                                  # Choose the first peak as our pitch component lag
    pitch = sr / lag                                                                # Transform lag into frequency
    return pitch

# Define the possible values for the parameter
notes = ['c', 'd', 'e', 'f', 'g', 'a', 'b']
osc_types = ['sine', 'square', 'sawtooth', 'triangle']
frequencies = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]

# Generate and save n different sound files
for i in range(n):
    # Generate random values for each parameter
    note = random.choice(notes)
    osc = random.choice(osc_types)
    mod = random.randint(0, 10)
    attack = round(random.random(), 3)
    decay = round(random.random(), 3)
    sustain = round(random.random(), 3)
    release = round(random.random(), 3)
    cutoff = random.choice(frequencies)

    sound = synthesizer(note, osc, mod, attack, decay, sustain, release, cutoff)        # Call the synthesizer function with the generated parameter values
    sf.write(path / "{}.wav".format(i), sound, sr, 'PCM_16')                            # Write the sound to a WAV file in the audiodata folder
    param_values.append([note, osc, mod, attack, decay, sustain, release, cutoff])      # Add the input parameter values to the list

# Save the input parameter value list to a CSV file
with open(path / 'data.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["note", "osc", "mod", "attack", "decay", "sustain", "release", "cutoff"])
    csvwriter.writerows(param_values)