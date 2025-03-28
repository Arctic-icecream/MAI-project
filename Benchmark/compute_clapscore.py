from msclap import CLAP

def compute_clap_score(audio_path, text) -> float:
    """
    Computes the CLAP Score (cos similarity) between an audio and a text description.

    Parameters:
        audio_path (str): Path to the audio file.
        text (str): Text description generated by GPT4 to compare with the audio.

    Returns:
        float: CLAP similarity score between audio and text.
    """
    clap_model = CLAP(version='2023', use_cuda=False)
    text_embeddings = clap_model.get_text_embeddings([text])
    audio_embeddings = clap_model.get_audio_embeddings([audio_path])

    similarities = clap_model.compute_similarity(audio_embeddings, text_embeddings)

    return similarities[0][0]

text_caption = "Two dogs, one white and one black, standing on grass with a sidewalk in the background. The white dog appears to be inspecting or interacting with the smaller black dog."

score_origin = compute_clap_score("Benchmark/samples/original/81.wav", text_caption)
score_concat = compute_clap_score("Benchmark/samples/generated/81.wav", text_caption)

print(f"CLAP Score - Audio_origin: {score_origin}")
print(f"CLAP Score - Audio_concat: {score_concat}")

if score_concat > score_origin:
    print("Oh yeah! Audio_concat achieves a higher CLAP Score :))")
else:
    print("Sadly, Audio_concat did not improve the CLAP Score ://")
