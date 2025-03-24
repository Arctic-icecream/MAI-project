from pydub import AudioSegment
import os

def concatenate_audio_simple(audio_files, output_path):
    combined_audio = AudioSegment.empty()

    for file in audio_files:
        combined_audio += AudioSegment.from_file(file)  

    combined_audio.export(output_path, format="wav")

    return output_path


def concatenate_audio_with_silence(audio_files, output_path, silence_duration=300):
    combined_audio = AudioSegment.empty()

    silence = AudioSegment.silent(duration=silence_duration) 

    for file in audio_files:
        combined_audio += AudioSegment.from_file(file) + silence  

    combined_audio.export(output_path, format="wav")

    return output_path


def concatenate_audio_overplay(audio_files, output_path):
    base_audio = AudioSegment.from_file(audio_files[0])  # first audio as base audio

    for file in audio_files[1:]:
        audio = AudioSegment.from_file(file)
        base_audio = base_audio.overlay(audio)  # overplay audios

    base_audio.export(output_path, format="wav")

    return output_path


#audio_files = ["18_1.wav", "18_2.wav"]

#concatenate_audio_overplay(audio_files, output_path="/teamspace/studios/this_studio/Benchmark/eval_dataset/eval_audio/18.wav")


def mix_audio_files(audio_files, output_path):
    """
    Overlays multiple audio files with stereo panning, without adjusting volume.

    Parameters:
        audio_files (list): List of paths to audio files to be mixed.
        output_path (str): Path to save the final mixed audio.

    Returns:
        str: Path to the saved mixed audio file.
    """
    if not audio_files:
        raise ValueError("No audio files provided.")
    base_audio = AudioSegment.from_file(audio_files[0])

    # Automatically set stereo panning: center for first, then alternate left/right
    pan_values = [0.0]  # First audio remains centered
    if len(audio_files) == 2:
        pan_values.extend([-0.6, 0.6])  # One sound left, one right
    elif len(audio_files) == 3:
        pan_values.extend([-0.6, 0.0, 0.6])  # Left, center, right

    for i, file in enumerate(audio_files[1:], start=1):
        audio = AudioSegment.from_file(file)
        audio = audio.pan(float(pan_values[i]))  # Apply stereo panning

        base_audio = base_audio.overlay(audio)
 
    base_audio.export(output_path, format="wav")
    return output_path

def process_audio_folders(root_folder, output_folder):
    """
    Traverses all subfolders in the root directory, detects audio files, and mixes them.

    Parameters:
        root_folder (str): Path to the main directory containing subfolders with audio files.
        output_folder (str): Path to the directory where mixed audio files will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create the output directory if it does not exist

    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)
        if os.path.isdir(subdir_path):  # Check if it's a directory
            audio_files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith(".wav")]
            audio_files.sort()  # Sort files to maintain order

            if len(audio_files) > 1: 
                output_audio_path = os.path.join(output_folder, f"{subdir}.wav")
                mix_audio_files(audio_files, output_audio_path)
                print(f"Mixed audio saved: {output_audio_path}")
            else:
                print(f"Skipping {subdir}, not enough audio files to mix.")

root_folder = "/teamspace/studios/this_studio/Benchmark/eval_dataset/eval_audio_seg"
output_folder = "/teamspace/studios/this_studio/Benchmark/eval_dataset/mixed_audios"

process_audio_folders(root_folder, output_folder)


