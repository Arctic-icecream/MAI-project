import numpy as np
from scipy.linalg import sqrtm
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model

class FeatureExtractor:
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.eval()

    def extract_features(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        waveform = waveform.mean(dim=0)
        
        inputs = self.processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            features = self.model(inputs.input_values).last_hidden_state.mean(dim=1)
        
        features = features.squeeze().numpy()
        features = (features - np.mean(features)) / (np.std(features) + 1e-10)

        return features

def compute_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    sigma += np.eye(sigma.shape[0]) * 1e-6

    return mu, sigma

# Frechet Distance Caculate
def frechet_distance(mu_r, sigma_r, mu_g, sigma_g):
    mean_diff = np.sum((mu_r - mu_g) ** 2)

    sigma_sqrt, _ = sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(sigma_sqrt):
        sigma_sqrt = sigma_sqrt.real
    if np.isnan(sigma_sqrt).any() or np.isinf(sigma_sqrt).any():
        print("⚠ 警告: 矩阵计算溢出，使用备用方法！")
        sigma_sqrt = np.eye(sigma_r.shape[0])

    return mean_diff + np.trace(sigma_r + sigma_g - 2 * sigma_sqrt)

# FD Evaluation
def evaluate_fd(original_audio_paths, generated_audio_paths):
    extractor = FeatureExtractor()

    original_features = [extractor.extract_features(path) for path in original_audio_paths]
    mu_r, sigma_r = compute_statistics(original_features)

    generated_features = [extractor.extract_features(path) for path in generated_audio_paths]
    mu_g, sigma_g = compute_statistics(generated_features)

    fd_score = frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
    print(f"Frechet Distance (FD): {fd_score:.2f}")
    return fd_score

if __name__ == "__main__":
    original_audio_paths = ["/teamspace/studios/this_studio/Benchmark/samples/original/17.wav", "/teamspace/studios/this_studio/Benchmark/samples/original/18.wav","/teamspace/studios/this_studio/Benchmark/samples/original/45.wav","/teamspace/studios/this_studio/Benchmark/samples/original/65.wav","/teamspace/studios/this_studio/Benchmark/samples/original/74.wav","/teamspace/studios/this_studio/Benchmark/samples/original/81.wav"]
    generated_audio_paths = ["/teamspace/studios/this_studio/Benchmark/samples/generated/17.wav", "/teamspace/studios/this_studio/Benchmark/samples/generated/18.wav","/teamspace/studios/this_studio/Benchmark/samples/generated/45.wav","/teamspace/studios/this_studio/Benchmark/samples/generated/65.wav","/teamspace/studios/this_studio/Benchmark/samples/generated/74.wav","/teamspace/studios/this_studio/Benchmark/samples/generated/81.wav"

]

    evaluate_fd(original_audio_paths, generated_audio_paths)
