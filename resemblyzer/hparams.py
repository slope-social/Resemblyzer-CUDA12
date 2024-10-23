from pathlib import Path

librispeech_datasets = {
    "train": {
        "clean": ["LibriSpeech/train-clean-100", "LibriSpeech/train-clean-360"],
        "other": ["LibriSpeech/train-other-500"]
    },
    "test": {
        "clean": ["LibriSpeech/test-clean"],
        "other": ["LibriSpeech/test-other"]
    },
    "dev": {
        "clean": ["LibriSpeech/dev-clean"],
        "other": ["LibriSpeech/dev-other"]
    },
}

# Audio settings
sampling_rate = 16000
audio_norm_target_dBFS = -30
vad_window_length = 30
vad_moving_average_width = 8
vad_max_silence_length = 6

# Mel-filterbank settings
mel_window_length = 25
mel_window_step = 10
mel_n_channels = 40

# Model parameters
partials_n_frames = 160
model_hidden_size = 256
model_embedding_size = 256
model_num_layers = 3
