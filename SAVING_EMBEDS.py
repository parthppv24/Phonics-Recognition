# Program to save embeddings
#
import torch
import librosa
import numpy as np
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from scipy.spatial.distance import cosine
import os

# === CONFIGURATION ===

REFERENCE_AUDIO_PATHS = {
    "a": [r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\a.wav",
          r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\a_female_E_correct_7730.wav"],

    "b":[r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\b.wav",
        r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\b_female_E_correct_1135.wav"],

    "c":[r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\c.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\c_female_E_correct_8179.wav"],

    "d":[r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\d_female_E_correct_6322.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\d.wav"],

    "e":[r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\e.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\e_female_E_correct_2714.wav"],

    "f":[r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\f.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\CollectedAudio\f_female_A_correct_1918.wav"],

    "g":[r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\g.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\g_female_E_correct_4603.wav"],

    "h":[r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\h.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\h_female_E_correct_7784.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\h_female_E_correct_8968.wav"],

    "i":[r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\i.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\i_female_E_correct_1973.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\i_female_E_correct_4425.wav"],

    "j":[r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\j.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NEW_TEST_AUDIO\j_female_A_correct_8737.wav"],

    "k":[r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\k.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NEW_TEST_AUDIO\k_female_A_correct_4870.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NEW_TEST_AUDIO\k_female_A_correct_8423.wav"],

    "l":[r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\l.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\l_female_E_correct_4639.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\l_female_E_correct_7501.wav"],

    "m":[r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\m.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\m_female_E_correct_1860.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\m_female_E_correct_4896.wav"],

    "n":[r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\n.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\n_female_E_correct_3857.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\n_female_E_correct_4600.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\n_female_E_correct_4646.wav",],

    "o":[r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\o.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\o_female_E_correct_9981.wav"],

    "p":[r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\p.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\p_female_E_correct_7711.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\p_female_E_correct_8644.wav"],

    "q":[r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\q.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\q_female_E_correct_8846.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\q_female_E_correct_9099.wav"],

    "r":[r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\r.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NEW_TEST_AUDIO\r_female_A_correct_1815.wav"],

    "s":[r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\s.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\s_female_E_correct_3130.wav"],

    "t":[r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\t_female_E_correct_5970.wav"],

    "u":[r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\u.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\u_female_E_correct_1644.wav"],

    "v":[r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\v.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\v_female_E_correct_3555.wav"],

    "w":[r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\w.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\w_female_E_correct_5318.wav"],

    "x":[r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\x_female_E_correct_7857.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\x.wav"],

    "y":[r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\y.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\y_female_E_correct_5477.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\y_female_E_correct_5707.wav"],

    "z":[r"C:\Users\parth\Downloads\PRAC\Phonics\ref audio\z.wav",
         r"C:\Users\parth\Downloads\PRAC\Phonics\NewRefAudio\z_female_E_correct_2323.wav"]

    # Add other letters and their fixed reference files...
}

THRESHOLDS = {
    #"z": 0.43,
    "a":0.45,
    "b":0.60,
    "c":0.39,
    "d":0.41,
    "e":0.57,
    "f":0.44,
    "g":0.50,
    "h":0.43,
    "i":0.56,
    "j":0.70,
    "k":0.42,
    "l":0.37,
    "m":0.67,
    "n":0.50,
    "o":0.50,
    "p":0.47,
    "q":0.48,
    "r":0.50,
    "s":0.50,
    "t":0.39,
    "u":0.60,
    "v":0.48,
    "w":0.58,
    "x":0.50,
    "y":0.65,
    "z":0.60
# Add thresholds for other letters
}

# === GLOBAL VARIABLES ===

MODEL = None
FEATURE_EXTRACTOR = None
REFERENCE_EMBEDDINGS = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDINGS_DIR = 'embeddings'  # Directory to save embeddings


# === AUDIO HELPERS ===

def trim_silence(audio_np, sr, top_db=20):
    trimmed_audio, _ = librosa.effects.trim(audio_np, top_db=top_db)
    return trimmed_audio


def rms_normalize(audio_np):
    rms = np.sqrt(np.mean(audio_np ** 2))
    return audio_np / (rms + 1e-6)


def preprocess_audio(path, target_sr=16000):
    audio_np, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=target_sr)
    audio_np = trim_silence(audio_np, target_sr)
    audio_np = rms_normalize(audio_np)
    return torch.tensor(audio_np, dtype=torch.float32)


# === SETUP FUNCTION (RUN BEFORE GAME STARTS) ===

def preload_model_and_refs():
    global MODEL, FEATURE_EXTRACTOR, REFERENCE_EMBEDDINGS

    print("📦 Loading model and feature extractor...")
    MODEL = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(DEVICE)
    FEATURE_EXTRACTOR = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")

    print("📁 Processing reference audios and saving embeddings...")
    if not os.path.exists(EMBEDDINGS_DIR):
        os.makedirs(EMBEDDINGS_DIR)

    for letter, paths in REFERENCE_AUDIO_PATHS.items():
        embeddings = []
        for path in paths:
            audio_tensor = preprocess_audio(path)
            emb = get_embedding(audio_tensor)
            embeddings.append(emb)
        # Save embeddings for each letter as a .npy file
        embeddings_path = os.path.join(EMBEDDINGS_DIR, f'{letter}_embeddings.npy')
        np.save(embeddings_path, np.array(embeddings))
        REFERENCE_EMBEDDINGS[letter] = embeddings
    print("✅ Model and references loaded and embeddings saved successfully.\n")


# === EMBEDDING FUNCTION ===

def get_embedding(audio_tensor):
    features = FEATURE_EXTRACTOR(audio_tensor.numpy(), sampling_rate=16000, return_tensors="pt")
    input_values = features.input_values.to(DEVICE)
    with torch.no_grad():
        outputs = MODEL(input_values)
    hidden_states = outputs.last_hidden_state
    embedding = hidden_states.mean(dim=1).squeeze(0).cpu().numpy().flatten()
    return embedding


# === RUNTIME FUNCTION ===

# def is_pronunciation_correct(input_audio_path, letter):
#     if letter not in REFERENCE_EMBEDDINGS or letter not in THRESHOLDS:
#         raise ValueError(f"Missing data for letter '{letter}'")
#
#     input_audio = preprocess_audio(input_audio_path)
#     input_emb = get_embedding(input_audio)
#
#     similarities = [1 - cosine(ref_emb, input_emb) for ref_emb in REFERENCE_EMBEDDINGS[letter]]
#     avg_similarity = np.mean(similarities)
#
#     return avg_similarity > THRESHOLDS[letter]


# === FUNCTION TO LOAD EMBEDDINGS ===

def load_embeddings():
    global REFERENCE_EMBEDDINGS
    print("🔄 Loading saved embeddings...")
    for letter in REFERENCE_AUDIO_PATHS.keys():
        embeddings_path = os.path.join(EMBEDDINGS_DIR, f'{letter}_embeddings.npy')
        if os.path.exists(embeddings_path):
            embeddings = np.load(embeddings_path)
            REFERENCE_EMBEDDINGS[letter] = embeddings
    print("✅ Embeddings loaded successfully.\n")

def inspect_all_embeddings(embeddings_dir=EMBEDDINGS_DIR):
    """
    Loads and prints all .npy embeddings stored in the embeddings directory.
    Displays the letter and the shape of each embedding array.
    """
    if not os.path.exists(embeddings_dir):
        print(f"❌ Embeddings directory '{embeddings_dir}' not found.")
        return

    print(f"🔍 Inspecting embeddings in '{embeddings_dir}'...\n")

    files = [f for f in os.listdir(embeddings_dir) if f.endswith('.npy')]
    if not files:
        print("⚠️ No .npy embedding files found.")
        return

    for file in sorted(files):
        try:
            embedding = np.load(os.path.join(embeddings_dir, file))
            print(f"✅ {file} — shape: {embedding.shape}\n{embedding}\n")
        except Exception as e:
            print(f"❌ Failed to load {file}: {e}")


if __name__ == "__main__":
    # Preload model and save embeddings
    inspect_all_embeddings()
    # Example usage of the saved embeddings
    load_embeddings()

