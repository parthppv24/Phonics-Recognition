# FastAPI implementation using pre-saved embeddings

import os
import torch
import librosa
import numpy as np
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from scipy.spatial.distance import cosine
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
from typing import Dict, List, Optional, Any

# === CONFIGURATION ===

# Thresholds for each letter
THRESHOLDS = {
    "a": 0.45, "b": 0.60, "c": 0.39, "d": 0.41, "e": 0.57, "f": 0.44, "g": 0.50,
    "h": 0.43, "i": 0.56, "j": 0.70, "k": 0.42, "l": 0.37, "m": 0.67, "n": 0.50,
    "o": 0.50, "p": 0.47, "q": 0.48, "r": 0.50, "s": 0.50, "t": 0.39, "u": 0.60,
    "v": 0.48, "w": 0.58, "x": 0.50, "y": 0.65, "z": 0.60
}

# Directory where embeddings are stored
EMBEDDINGS_DIR = r'C:\Users\parth\Downloads\PRAC\Phonics\ProdCODE\embeddings'


# === PYDANTIC MODELS ===
class PronunciationResponse(BaseModel):
    letter: str
    similarity: float
    threshold: float
    correct: bool


class HealthResponse(BaseModel):
    status: str
    letters_available: List[str]
    device: str
    embeddings_count: Dict[str, int]


class ErrorResponse(BaseModel):
    error: str


# === MODEL INITIALIZATION ===
class ModelManager:
    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.reference_embeddings = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_initialized = False

    def initialize(self):
        if self.is_initialized:
            return True

        print("📦 Loading model and feature extractor...")
        try:
            self.model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(self.device)
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
            print("✅ Model loaded successfully")

            # Load pre-saved embeddings
            self.is_initialized = self.load_embeddings()
            if not self.is_initialized:
                print(
                    "❌ Failed to load embeddings. Please check if embeddings directory exists and contains .npy files.")

            return self.is_initialized
        except Exception as e:
            print(f"❌ Error during initialization: {str(e)}")
            return False

    def load_embeddings(self):
        print("🔄 Loading saved embeddings...")

        # Check if embeddings directory exists
        if not os.path.exists(EMBEDDINGS_DIR):
            print(f"❌ Embeddings directory '{EMBEDDINGS_DIR}' not found.")
            return False

        # Get all .npy files in the directory
        embedding_files = [f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith('_embeddings.npy')]

        if not embedding_files:
            print("⚠️ No embedding files found in the directory.")
            return False

        # Load each embedding file
        for file in embedding_files:
            try:
                # Extract letter from filename (assuming format like 'a_embeddings.npy')
                letter = file.split('_')[0]

                # Load the embeddings
                embeddings_path = os.path.join(EMBEDDINGS_DIR, file)
                embeddings = np.load(embeddings_path)

                # Store in our dictionary
                self.reference_embeddings[letter] = embeddings
                print(f"✅ Loaded embeddings for letter '{letter}' - Shape: {embeddings.shape}")
            except Exception as e:
                print(f"❌ Failed to load embeddings from {file}: {e}")

        print(f"✅ Successfully loaded embeddings for {len(self.reference_embeddings)} letters")
        return len(self.reference_embeddings) > 0

    def get_embedding(self, audio_tensor):
        # Fix: Make sure audio_tensor is 1D before converting to numpy
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze()

        # Ensure we're working with float32
        audio_tensor = audio_tensor.to(torch.float32)

        # Fix: Handle empty tensors
        if audio_tensor.numel() == 0:
            raise ValueError("Audio tensor is empty")

        # Fix: Convert to NumPy safely
        audio_np = audio_tensor.numpy()

        features = self.feature_extractor(audio_np, sampling_rate=16000, return_tensors="pt")
        input_values = features.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)

        hidden_states = outputs.last_hidden_state
        embedding = hidden_states.mean(dim=1).squeeze(0).cpu().numpy().flatten()
        return embedding


# Create a singleton model manager
model_manager = ModelManager()


# === AUDIO PROCESSING FUNCTIONS ===
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


# === DEPENDENCY ===
def get_model_manager():
    """Dependency to ensure model is initialized before handling requests"""
    if not model_manager.is_initialized:
        if not model_manager.initialize():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not initialized. Please try again later."
            )
    return model_manager


# === FASTAPI APP ===
app = FastAPI(
    title="Pronunciation Checker API",
    description="API to check pronunciation of letters using WavLM embeddings",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    # Use background task to initialize model asynchronously
    background_tasks = BackgroundTasks()
    background_tasks.add_task(model_manager.initialize)


@app.post(
    "/check-pronunciation",
    response_model=PronunciationResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    }
)
async def check_pronunciation(
        audio: UploadFile = File(...),
        letter: str = Form(...),
        model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Check pronunciation of a letter based on audio input.

    - **audio**: WAV audio file of pronunciation
    - **letter**: The letter being pronounced
    """
    letter = letter.lower()

    # Validate letter
    if letter not in model_manager.reference_embeddings:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No reference embeddings for letter '{letter}'"
        )

    if letter not in THRESHOLDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No threshold defined for letter '{letter}'"
        )

    # Process audio
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            contents = await audio.read()
            temp_file.write(contents)
            temp_path = temp_file.name

        # Process the audio file
        input_audio = preprocess_audio(temp_path)
        input_emb = model_manager.get_embedding(input_audio)

        # Calculate similarity with all reference embeddings for this letter
        similarities = [1 - cosine(ref_emb, input_emb) for ref_emb in model_manager.reference_embeddings[letter]]
        avg_similarity = float(np.mean(similarities))
        threshold = THRESHOLDS[letter]
        match = avg_similarity > threshold

        # Return result
        return PronunciationResponse(
            letter=letter,
            similarity=avg_similarity,
            threshold=threshold,
            correct=match
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing audio: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": ErrorResponse}}
)
async def health_check(model_manager: ModelManager = Depends(get_model_manager)):
    """Check if the API is ready to process requests"""
    return HealthResponse(
        status="Ready",
        letters_available=list(model_manager.reference_embeddings.keys()),
        device=str(model_manager.device),
        embeddings_count={letter: len(embs) for letter, embs in model_manager.reference_embeddings.items()}
    )


if __name__ == "__main__":
    import uvicorn

    # Start server
    print("🚀 Starting FastAPI server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)