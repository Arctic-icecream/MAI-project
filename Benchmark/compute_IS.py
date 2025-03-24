import numpy as np
import openl3
import soundfile as sf
import os
import librosa
from scipy.special import softmax
import matplotlib.pyplot as plt

def get_audio_files(folder_path):
    audio_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))
    if not audio_files:
        raise ValueError(f"Wrong: files '{folder_path}' cannot find")
    return audio_files

# OpenL3. Classifier
class OpenL3Classifier:
    def __init__(self):
        self.model = openl3.models.load_audio_embedding_model(input_repr="mel128", embedding_size=512, content_type="env")

    def classify_audio(self, audio_path):
        try:
            waveform, sr = sf.read(audio_path)
            if sr != 48000:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=48000)
                sr = 48000
        except Exception as e:
            print(f"fail: {audio_path} - {str(e)}")
            return None

        embedding, _ = openl3.get_audio_embedding(waveform, sr, model=self.model, input_repr="mel128", content_type="env", embedding_size=512)

        if embedding is None or embedding.size == 0:
            print(f"warning: {audio_path} feature is empty, skip!")
            return None

        probabilities = softmax((embedding.mean(axis=0) - np.mean(embedding.mean(axis=0))) / (np.std(embedding.mean(axis=0)) + 1e-10)) 

        return probabilities

# Inception Score (IS) caculate
def inception_score(probabilities, splits=10):
    """
        Inception Score (IS)
        probabilities: shape (num_samples, num_classes)，
        splits: 10
    """
    num_samples = probabilities.shape[0]
    if num_samples < 2:
        raise ValueError(" Error: at least two files！")
    
    splits = min(splits, num_samples)
    split_size = max(1, num_samples // splits)

    scores = []
    for i in range(splits):
        subset = probabilities[i * split_size:(i + 1) * split_size]  
        p_yx = subset  
        p_y = np.mean(p_yx, axis=0, keepdims=True)  

       
        kl_div = p_yx * (np.log(p_yx + 1e-10) - np.log(p_y + 1e-10))
        kl_div = np.sum(kl_div, axis=1)  
        score = np.exp(np.mean(kl_div))  

        scores.append(score)

    return np.mean(scores), np.std(scores) 

# IS
def evaluate_is(folder_path):
    classifier = OpenL3Classifier()

    print(f"scan: {folder_path}...")
    audio_paths = get_audio_files(folder_path)

    print("Extracted audio category distribution...")
    probabilities = []
    for path in audio_paths:
        prob = classifier.classify_audio(path)
        if prob is not None:
            probabilities.append(prob)

    probabilities = np.array(probabilities)

    if probabilities.shape[0] < 2:
        raise ValueError(" Error: at least two files！")

    print("nception Score...")
    mean_is, std_is = inception_score(probabilities)

    print(f"\n finish: Inception Score (IS) = {mean_is:.2f} ± {std_is:.2f}")

    return mean_is, std_is

if __name__ == "__main__":
    folder_path = "/teamspace/studios/this_studio/Benchmark/eval_dataset/eval_audio1"  

    evaluate_is(folder_path)
