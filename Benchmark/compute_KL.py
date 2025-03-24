import numpy as np
import openl3
import soundfile as sf
from scipy.special import rel_entr

class OpenL3Classifier:
    def __init__(self):
        self.model = openl3.models.load_audio_embedding_model(input_repr="mel128", embedding_size=512, content_type="env")

    def classify_audio(self, audio_path):
        waveform, sr = sf.read(audio_path)
        embedding, _ = openl3.get_audio_embedding(waveform, sr, model=self.model, input_repr="mel128", content_type="env", embedding_size=512)

        if embedding.size == 0:
            print(f"Warning: {audio_path} extracted feature is empty, skip!")
            return None

        probabilities = embedding.mean(axis=0)
        probabilities /= np.sum(probabilities)  

        return probabilities

def kl_divergence(p, q):
    return np.sum(rel_entr(p, q))

def evaluate_kl(original_audio_paths, generated_audio_paths):
    classifier = OpenL3Classifier()

    print("Extracted raw audio category distribution...")
    original_probabilities = []
    for path in original_audio_paths:
        prob = classifier.classify_audio(path)
        if prob is not None:
            original_probabilities.append(prob)

    print("Extraction generates audio category distribution...")
    generated_probabilities = []
    for path in generated_audio_paths:
        prob = classifier.classify_audio(path)
        if prob is not None:
            generated_probabilities.append(prob)

    if len(original_probabilities) < 2 or len(generated_probabilities) < 2:
        raise ValueError("Error: At least 2 original and 2 generated audio are needed to calculate KL dispersion!")


    p_real = np.mean(original_probabilities, axis=0)
    p_gen = np.mean(generated_probabilities, axis=0)


    kl_score = kl_divergence(p_real, p_gen)

    print(f"\n Finish: Kullback-Leibler Divergence (KL) = {kl_score:.4f}")
    return kl_score

if __name__ == "__main__":
    original_audio_paths = ["/teamspace/studios/this_studio/Benchmark/samples/original/17.wav", "/teamspace/studios/this_studio/Benchmark/samples/original/18.wav","/teamspace/studios/this_studio/Benchmark/samples/original/45.wav","/teamspace/studios/this_studio/Benchmark/samples/original/65.wav","/teamspace/studios/this_studio/Benchmark/samples/original/74.wav","/teamspace/studios/this_studio/Benchmark/samples/original/81.wav"]
    generated_audio_paths = ["/teamspace/studios/this_studio/Benchmark/samples/generated/17.wav", "/teamspace/studios/this_studio/Benchmark/samples/generated/18.wav","/teamspace/studios/this_studio/Benchmark/samples/generated/45.wav","/teamspace/studios/this_studio/Benchmark/samples/generated/65.wav","/teamspace/studios/this_studio/Benchmark/samples/generated/74.wav","/teamspace/studios/this_studio/Benchmark/samples/generated/81.wav"]

    evaluate_kl(original_audio_paths, generated_audio_paths)
