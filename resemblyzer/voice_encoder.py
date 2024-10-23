# First, let's create the updated voice_encoder.py:

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import typing as t

class SpeakerEncoder(nn.Module):
    def __init__(self, device: t.Optional[torch.device] = None, verbose=True):
        super().__init__()
        
        # Define model architecture
        self.lstm = nn.LSTM(input_size=40,
                           hidden_size=256, 
                           num_layers=3,
                           batch_first=True)
        self.linear = nn.Linear(in_features=256, 
                              out_features=256)
        self.relu = nn.ReLU()

        # Set device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if verbose:
            print(f'Using device: {self.device}')
            if self.device.type == 'cuda':
                print(f'CUDA version: {torch.version.cuda}')
        
        # Move model to device
        self.to(self.device)
        
    def do_gradient_ops(self):
        # Gradient scale
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
            
        # Gradient clipping
        clip_grad_norm_(self.parameters(), 3, norm_type=2)
    
    def forward(self, utterances: torch.tensor, hidden_init: t.Optional[torch.tensor] = None):
        """
        Computes the embeddings of a batch of utterance spectrograms.
        
        :param utterances: batch of mel-scale filterbanks of same duration as a tensor (batch_size, n_frames, n_channels) 
        :param hidden_init: initial hidden state of the LSTM as a tensor of shape (num_layers, batch_size, hidden_size)
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """
        # Pass the input through the LSTM layers and retrieve all outputs, the final hidden state
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        
        # We take only the hidden state of the last layer
        embeds_raw = self.relu(self.linear(hidden[-1]))
        
        # L2-normalize it
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)
        
        return embeds

    @staticmethod
    def compute_partial_slices(n_samples: int, rate: float, min_coverage: float):
        """
        Computes where to split an utterance waveform and its corresponding mel spectrogram to obtain 
        partial utterances of <min_coverage> seconds with <rate> partial utterances per second of audio.
        """
        assert 0 < min_coverage <= 1
        
        # Compute how many partial utterances we'll make
        total_duration = n_samples / sampling_rate
        n_partials = int(np.ceil(total_duration * rate))
        
        # Compute the duration of each partial
        partial_duration = min_coverage / rate
        partial_n_samples = int(np.ceil(sampling_rate * partial_duration))
        coverage = partial_n_samples / sampling_rate
        
        # Compute the step between partial utterances
        step_duration = total_duration / n_partials
        step = int(np.ceil(sampling_rate * step_duration))
        
        # Split the waveform into partials
        wave_slices = []
        mel_slices = []
        for offset in range(0, n_samples, step):
            wave_slice = slice(offset, offset + partial_n_samples)
            mel_slice = slice(offset // hop_length, (offset + partial_n_samples) // hop_length)
            if mel_slice.stop - mel_slice.start <= minimum_frames:
                break
            wave_slices.append(wave_slice)
            mel_slices.append(mel_slice)
            
        return wave_slices, mel_slices

class VoiceEncoder:
    def __init__(self, weights_fpath: t.Optional[str] = None, device: t.Optional[torch.device] = None, verbose=True):
        """
        :param weights_fpath: path to saved model weights
        :param device: either a torch device or the name of a torch device (e.g. "cpu", "cuda")
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = SpeakerEncoder(self.device, verbose)
        
        # Load model weights
        weights_fpath = weights_fpath or Path(__file__).resolve().parent.joinpath("pretrained.pt")
        try:
            checkpoint = torch.load(weights_fpath, map_location=self.device, weights_only=True)
            consume_prefix_in_state_dict_if_present(checkpoint, "module.")
            self.model.load_state_dict(checkpoint)
            self.model.eval()
        except Exception as e:
            print(f"Error loading model weights from {weights_fpath}: {str(e)}")
            print("Attempting to continue with uninitialized model...")

    def embed_utterance(self, wav: np.ndarray, return_partials=False, rate=1.3, min_coverage=0.75):
        """
        Computes an embedding for a single utterance.
        
        # Arguments
        wav: Preprocessed waveform of a single utterance
        return_partials: if True, the partial embeddings will also be returned
        rate: how many partial utterances to generate per second of audio.
        min_coverage: when generating partial utterances, some utterances may be too short. 
        
        # Returns
        The embedding as a numpy array of float32 of shape (model_embedding_size,).
        """
        # Compute where to split the utterance into partials
        wave_slices, mel_slices = self.model.compute_partial_slices(len(wav), rate, min_coverage)
        max_wave_length = wave_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
        
        # Split the utterance into partials and forward them through the model
        mel = wav2mel(wav)
        mels = np.array([mel[s] for s in mel_slices])
        with torch.no_grad():
            mels = torch.from_numpy(mels).to(self.device)
            partial_embeds = self.model(mels).cpu().numpy()
        
        # Compute the utterance embedding from the partial embeddings
        raw_embed = np.mean(partial_embeds, axis=0)
        embed = raw_embed / np.linalg.norm(raw_embed, 2)
        
        if return_partials:
            return embed, partial_embeds, wave_slices
        return embed

    def embed_speaker(self, wavs: t.List[np.ndarray], **kwargs):
        """
        Compute the embedding of a collection of wavs (presumably from the same speaker) by averaging.
        """
        raw_embed = np.mean([self.embed_utterance(wav, return_partials=False, **kwargs) 
                          for wav in wavs], axis=0)
        return raw_embed / np.linalg.norm(raw_embed, 2)

    def similarity_matrix(self, embeds):
        """
        Computes the similarity matrix according the section 2.1 of GE2E.

        :param embeds: the embeddings as a tensor of shape (batch_size, embedding_size)
        :return: the similarity matrix as a tensor of shape (batch_size, batch_size)
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # Inclusive centroids (1 per speaker)
        centroids_incl = torch.mean(embeds, dim=1)
        centroids_incl = centroids_incl / torch.norm(centroids_incl, dim=1, keepdim=True)
        
        # Exclusive centroids (1 per utterance)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl / torch.norm(centroids_excl, dim=2, keepdim=True)

        # Similarity matrix
        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                               speakers_per_batch)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        
        return sim_matrix

    def loss(self, embeds):
        """
        Computes the loss according to section 2.1 of GE2E.
        
        :param embeds: the embeddings as a tensor of shape (batch_size, embedding_size)
        :return: the loss and the EER for this batch of embeddings.
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # Loss
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        loss = torch.mean(
            torch.cat([
                -(F.log_softmax(sim_matrix[speakers_per_batch:, :, i], dim=1)[:, i] +
                  F.log_softmax(sim_matrix[i, :, :speakers_per_batch].T, dim=1)[:, i])
                for i in range(speakers_per_batch)
            ])
        )
        
        # EER (not backpropagated)
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
            labels = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
            preds = sim_matrix.detach().cpu().numpy().reshape((speakers_per_batch * utterances_per_speaker, speakers_per_batch))
            fpr, tpr, thresholds = roc_curve(labels, preds)
            eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
        
        return loss, eer
