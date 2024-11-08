import numpy as np
from resemblyzer.hparams import *
from resemblyzer import audio, preprocess_wav
from sklearn.cluster import AgglomerativeClustering
from typing import List, Tuple, Union
import torch
import torch.nn.utils.rnn as rnn
import logging
from pathlib import Path
from time import perf_counter as timer
import torch.nn as nn
from functools import wraps

logger = logging.getLogger(__name__)

def ensure_cpu_tensor(func):
    """Decorator to ensure tensors are moved to CPU before numpy conversion"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if torch.is_tensor(result) and result.is_cuda:
            return result.detach().cpu()
        return result
    return wrapper

class CUDAVoiceEncoder(nn.Module):
    def __init__(self, device: Union[str, torch.device]=None, verbose=True, weights_fpath: Union[Path, str]=None):
        """
        Enhanced VoiceEncoder with proper CUDA handling.
        """
        super().__init__()

        # Define the network
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

        # Get the target device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Load the pretrained model weights
        if weights_fpath is None:
            weights_fpath = Path(__file__).resolve().parent.joinpath("pretrained.pt")
        else:
            weights_fpath = Path(weights_fpath)

        if not weights_fpath.exists():
            raise Exception(f"Couldn't find the voice encoder pretrained model at {weights_fpath}")
        
        start = timer()
        try:
            checkpoint = torch.load(weights_fpath, map_location=device, weights_only=True)
            if "model_state" in checkpoint:
                self.load_state_dict(checkpoint["model_state"], strict=False)
            else:
                self.load_state_dict(checkpoint, strict=False)
        except RuntimeError as e:
            logger.error(f"Error loading model weights: {str(e)}")
            raise

        self.to(device)
        if verbose:
            logger.info(f"Loaded the voice encoder model on {device.type} in {timer() - start:.2f} seconds.")

    def forward(self, mels: torch.FloatTensor):
        """
        CUDA-aware forward pass.
        """
        with torch.cuda.device(self.device):
            _, (hidden, _) = self.lstm(mels)
            embeds_raw = self.relu(self.linear(hidden[-1]))
            return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    @staticmethod
    def compute_partial_slices(n_samples: int, rate, min_coverage):
        """
        Compute partial utterance slices.
        """
        assert 0 < min_coverage <= 1

        samples_per_frame = int((sampling_rate * mel_window_step / 1000))
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        frame_step = int(np.round((sampling_rate / rate) / samples_per_frame))
        
        if frame_step <= 0:
            raise ValueError("The rate is too high")
        if frame_step > partials_n_frames:
            raise ValueError(f"The rate is too low, it should be {sampling_rate / (samples_per_frame * partials_n_frames)} at least")

        wav_slices, mel_slices = [], []
        steps = max(1, n_frames - partials_n_frames + frame_step + 1)
        for i in range(0, steps, frame_step):
            mel_range = np.array([i, i + partials_n_frames])
            wav_range = mel_range * samples_per_frame
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))

        if len(mel_slices) > 1:
            last_wav_range = wav_slices[-1]
            coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
            if coverage < min_coverage:
                mel_slices = mel_slices[:-1]
                wav_slices = wav_slices[:-1]

        return wav_slices, mel_slices

    def embed_utterance(self, wav: Union[np.ndarray, torch.Tensor], return_partials=False, rate=1.3, min_coverage=0.75):
        """
        CUDA-aware utterance embedding.
        """
        # Handle CUDA tensor input
        if torch.is_tensor(wav):
            if wav.is_cuda:
                wav = wav.cpu()
            wav = wav.detach().numpy()

        # Ensure correct dimensions
        if len(wav.shape) == 2:
            wav = wav[0]
        elif len(wav.shape) > 2:
            raise ValueError(f"Waveform has too many dimensions ({len(wav.shape)})")

        # Compute slices and pad if necessary
        wav_slices, mel_slices = self.compute_partial_slices(len(wav), rate, min_coverage)
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

        # Process the utterance
        mel = audio.wav_to_mel_spectrogram(wav)
        mels = np.array([mel[s] for s in mel_slices])
        
        with torch.no_grad(), torch.cuda.device(self.device):
            mels = torch.from_numpy(mels).to(self.device)
            partial_embeds = self(mels)
            partial_embeds = partial_embeds.cpu().numpy()

        raw_embed = np.mean(partial_embeds, axis=0)
        embed = raw_embed / np.linalg.norm(raw_embed, 2)

        return (embed, partial_embeds, wav_slices) if return_partials else embed

class ResemblyzerDiarizer:
    def __init__(self, use_cuda: bool = True):
        """Initialize with optimal device management"""
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        try:
            self.encoder = CUDAVoiceEncoder(device=self.device)
            logger.info(f"Initialized CUDA-aware VoiceEncoder on device: {self.device}")
        except Exception as e:
            logger.warning(f"Failed to initialize VoiceEncoder on {self.device}: {str(e)}")
            self.device = torch.device("cpu")
            self.encoder = CUDAVoiceEncoder(device=self.device)

    def preprocess_audio(self, audio_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Preprocess audio data with CUDA awareness"""
        if isinstance(audio_data, np.ndarray):
            audio_tensor = torch.from_numpy(audio_data)
        elif isinstance(audio_data, torch.Tensor):
            audio_tensor = audio_data
        else:
            raise TypeError(f"Unsupported audio data type: {type(audio_data)}")
        
        return audio_tensor.to(self.device, non_blocking=True)

    def process_batch(self, segments: List[dict], audio_tensor: torch.Tensor) -> np.ndarray:
        """Process multiple segments in batch for improved efficiency"""
        with torch.no_grad(), torch.cuda.device(self.device):
            batch_embeddings = []
            batch_size = 32  # Adjustable based on GPU memory

            for i in range(0, len(segments), batch_size):
                batch = segments[i:i + batch_size]
                try:
                    batch_audio = [audio_tensor[int(s['start'] * 16000):int(s['end'] * 16000)].contiguous() 
                                 for s in batch]
                    
                    batch_audio = [audio for audio in batch_audio if audio.numel() > 0]
                    if not batch_audio:
                        continue

                    max_len = max(audio.size(0) for audio in batch_audio)
                    padded_audio = rnn.pad_sequence(
                        batch_audio, batch_first=True, padding_value=0
                    )
                    
                    embeddings = self.encoder.embed_utterance(padded_audio)
                    batch_embeddings.append(embeddings)

                except Exception as e:
                    logger.error(f"Failed to process batch: {str(e)}", exc_info=True)
                    continue

            if not batch_embeddings:
                return np.zeros((1, 256), dtype=np.float32)

            return np.concatenate(batch_embeddings, axis=0)

    def _extract_embeddings(self, audio: np.ndarray, segments: List[dict]) -> np.ndarray:
        """Extract embeddings with optimized tensor handling"""
        embeddings = []
        
        try:
            audio_tensor = self.preprocess_audio(audio)
            
            if len(segments) > 32:
                return self.process_batch(segments, audio_tensor)
            
            for segment in segments:
                start_sample = int(segment['start'] * 16000)
                end_sample = int(segment['end'] * 16000)
                
                try:
                    segment_audio = audio_tensor[start_sample:end_sample].contiguous()
                    
                    if segment_audio.numel() > 0:
                        with torch.cuda.device(self.device):
                            with torch.no_grad():
                                embedding = self.encoder.embed_utterance(segment_audio)
                                if embedding is not None:
                                    embeddings.append(embedding)
                    else:
                        logger.warning(f"Empty segment detected: {segment}")
                        
                except Exception as e:
                    logger.warning(f"Failed to process segment: {str(e)}", exc_info=True)
                    continue

        except Exception as e:
            logger.error(f"Failed during embedding extraction: {str(e)}", exc_info=True)
            return np.zeros((1, 256), dtype=np.float32)

        if not embeddings:
            logger.warning("No valid embeddings extracted")
            return np.zeros((1, 256), dtype=np.float32)

        try:
            return np.stack(embeddings)
        except Exception as e:
            logger.error(f"Failed to stack embeddings: {str(e)}", exc_info=True)
            return np.zeros((1, 256), dtype=np.float32)

    def _perform_clustering(self, embeddings: np.ndarray, min_speakers: int, max_speakers: int) -> np.ndarray:
        """Perform speaker clustering with constraints"""
        try:
            # Initial clustering with agglomerative clustering
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.3,
                linkage='average'
            )
            labels = clustering.fit_predict(embeddings)

            # Adjust number of speakers if needed
            unique_speakers = np.unique(labels)
            n_speakers = len(unique_speakers)

            if n_speakers < min_speakers:
                logger.info(f"Found {n_speakers} speakers, forcing minimum of {min_speakers}")
                clustering = AgglomerativeClustering(n_clusters=min_speakers)
                labels = clustering.fit_predict(embeddings)
            elif n_speakers > max_speakers:
                logger.info(f"Found {n_speakers} speakers, limiting to maximum of {max_speakers}")
                clustering = AgglomerativeClustering(n_clusters=max_speakers)
                labels = clustering.fit_predict(embeddings)

            return labels

        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}", exc_info=True)
            return np.zeros(len(embeddings), dtype=np.int32)

    def _merge_adjacent_segments(self, segments: List[dict]) -> List[dict]:
        """Merge adjacent segments from same speaker"""
        if not segments:
            return []

        try:
            merged_segments = []
            current_segment = segments[0].copy()

            for segment in segments[1:]:
                # Check if current segment and next segment have same speaker and are adjacent
                if (current_segment['speaker'] == segment['speaker'] and 
                    abs(current_segment['end'] - segment['start']) < 0.5):  # 500ms threshold
                    # Merge segments
                    current_segment['end'] = segment['end']
                    # Merge text if present
                    if 'text' in current_segment and 'text' in segment:
                        current_segment['text'] = (current_segment['text'].strip() + ' ' + 
                                                 segment['text'].strip()).strip()
                else:
                    # Add current segment to merged list and start new one
                    merged_segments.append(current_segment)
                    current_segment = segment.copy()

            # Add final segment
            merged_segments.append(current_segment)

            return merged_segments

        except Exception as e:
            logger.error(f"Segment merging failed: {str(e)}", exc_info=True)
            return segments

    def diarize(self, audio: np.ndarray, segments: List[dict], min_speakers: int = 2, max_speakers: int = 10) -> List[dict]:
        """Main diarization method with improved error handling"""
        try:
            # Create smaller segments if necessary
            max_segment_duration = 30
            new_segments = []
            
            for segment in segments:
                start = segment['start']
                end = segment['end']
                while start < end:
                    new_end = min(start + max_segment_duration, end)
                    new_segments.append({
                        'start': start,
                        'end': new_end,
                        'text': segment.get('text', ''),
                        'words': segment.get('words', []),
                        'speaker': segment.get('speaker', None)
                    })
                    start = new_end

            # Extract embeddings with proper error handling
            embeddings = self._extract_embeddings(audio, new_segments)
            logger.info(f"Number of embeddings: {len(embeddings)}")

            if len(embeddings) < 2:
                logger.warning("Not enough embeddings for clustering. Assigning all segments to the same speaker.")
                for segment in new_segments:
                    segment['speaker'] = "SPEAKER_0"
                return new_segments

            # Perform clustering with proper error handling
            try:
                labels = self._perform_clustering(embeddings, min_speakers, max_speakers)
                logger.info(f"Number of unique speakers detected: {len(np.unique(labels))}")
            except Exception as e:
                logger.error(f"Clustering failed, defaulting to single speaker: {str(e)}")
                labels = np.zeros(len(new_segments))

            # Assign speakers to segments
            for i, segment in enumerate(new_segments):
                segment['speaker'] = f"SPEAKER_{labels[i]}"

            # Handle word-level speaker assignment
            for segment in new_segments:
                if 'words' in segment:
                    for word in segment['words']:
                        word['speaker'] = segment['speaker']

            # Merge adjacent segments with improved error handling
            try:
                merged_segments = self._merge_adjacent_segments(new_segments)
                logger.info(f"Number of segments after merging: {len(merged_segments)}")
            except Exception as e:
                logger.error(f"Segment merging failed, using unmerged segments: {str(e)}")
                merged_segments = new_segments

            return merged_segments

        except Exception as e:
            logger.error(f"Diarization failed: {str(e)}", exc_info=True)
            # Return the original segments with a default speaker in case of failure
            for segment in segments:
                segment['speaker'] = "SPEAKER_0"
            return segments


def diarize(audio: np.ndarray, result: dict, min_speakers: int = 2, max_speakers: int = 10, use_cuda: bool = True) -> dict:
    """Main diarization interface with WaveTrace compatibility"""
    try:
        diarizer = ResemblyzerDiarizer(use_cuda=use_cuda)
        diarized_segments = diarizer.diarize(audio, result['segments'], min_speakers, max_speakers)
        result['segments'] = diarized_segments
        logger.info(f"Diarization complete. Number of segments: {len(result['segments'])}")
        logger.debug(f"First segment speaker: {result['segments'][0].get('speaker', 'No speaker')}")
        return result
    except Exception as e:
        logger.error(f"Diarization failed: {str(e)}", exc_info=True)
        return result

# Alias CUDAVoiceEncoder as VoiceEncoder for backward compatibility
VoiceEncoder = CUDAVoiceEncoder

__all__ = ['VoiceEncoder', 'CUDAVoiceEncoder']
