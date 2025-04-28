import os
import torch
import numpy as np
import traceback
import logging
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from scipy.spatial.distance import cosine
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app_errors.log")
    ]
)
logger = logging.getLogger("pronunciation_api")

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
    details: Optional[str] = None
    error_type: Optional[str] = None
    error_location: Optional[str] = None


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

        logger.info("📦 Loading model and feature extractor...")
        try:
            self.model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(self.device)
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
            logger.info("✅ Model loaded successfully")

            # Load pre-saved embeddings
            self.is_initialized = self.load_embeddings()
            if not self.is_initialized:
                logger.error(
                    "❌ Failed to load embeddings. Please check if embeddings directory exists and contains .npy files.")

            return self.is_initialized
        except Exception as e:
            logger.error(f"❌ Error during initialization: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def load_embeddings(self):
        logger.info("🔄 Loading saved embeddings...")

        # Check if embeddings directory exists
        if not os.path.exists(EMBEDDINGS_DIR):
            logger.error(f"❌ Embeddings directory '{EMBEDDINGS_DIR}' not found.")
            return False

        # Get all .npy files in the directory
        embedding_files = [f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith('_embeddings.npy')]

        if not embedding_files:
            logger.warning("⚠️ No embedding files found in the directory.")
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
                logger.info(f"✅ Loaded embeddings for letter '{letter}' - Shape: {embeddings.shape}")
            except Exception as e:
                logger.error(f"❌ Failed to load embeddings from {file}: {e}")
                logger.error(traceback.format_exc())

        logger.info(f"✅ Successfully loaded embeddings for {len(self.reference_embeddings)} letters")
        return len(self.reference_embeddings) > 0

    def get_embedding(self, audio_tensor):
        try:
            # Fix: Make sure audio_tensor is 1D before converting to numpy
            if audio_tensor.dim() > 1:
                logger.debug(f"Squeezing audio tensor from shape {audio_tensor.shape}")
                audio_tensor = audio_tensor.squeeze()

            # Ensure we're working with float32
            audio_tensor = audio_tensor.to(torch.float32)

            # Fix: Handle empty tensors
            if audio_tensor.numel() == 0:
                logger.error("Audio tensor is empty")
                raise ValueError("Audio tensor is empty")

            # Log tensor stats for debugging
            logger.debug(
                f"Audio tensor shape: {audio_tensor.shape}, min: {audio_tensor.min()}, max: {audio_tensor.max()}")

            # Fix: Convert to NumPy safely
            audio_np = audio_tensor.numpy()

            features = self.feature_extractor(audio_np, sampling_rate=16000, return_tensors="pt")
            input_values = features.input_values.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_values)

            hidden_states = outputs.last_hidden_state
            embedding = hidden_states.mean(dim=1).squeeze(0).cpu().numpy().flatten()
            logger.debug(f"Generated embedding with shape: {embedding.shape}")
            return embedding
        except Exception as e:
            logger.error(f"Error in get_embedding: {str(e)}")
            logger.error(traceback.format_exc())
            raise


# Create a singleton model manager
model_manager = ModelManager()


# === AUDIO PROCESSING FUNCTIONS ===
def trim_silence(audio_np, sr, top_db=20):
    try:
        logger.debug(f"Trimming silence from audio, shape before: {audio_np.shape}")
        trimmed_audio, _ = librosa.effects.trim(audio_np, top_db=top_db)
        logger.debug(f"Shape after trimming: {trimmed_audio.shape}")
        return trimmed_audio
    except Exception as e:
        logger.error(f"Error in trim_silence: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def rms_normalize(audio_np):
    try:
        logger.debug(f"Normalizing audio, shape: {audio_np.shape}")
        rms = np.sqrt(np.mean(audio_np ** 2))
        if rms < 1e-6:
            logger.warning("Very low RMS value detected in audio, might be too quiet")
        return audio_np / (rms + 1e-6)
    except Exception as e:
        logger.error(f"Error in rms_normalize: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def preprocess_audio(path, target_sr=16000):
    try:
        logger.info(f"Loading audio from: {path}")

        # Check if file exists and is readable
        if not os.path.exists(path):
            logger.error(f"Audio file not found: {path}")
            raise FileNotFoundError(f"Audio file not found: {path}")

        # Use audioread instead of librosa.load
        import audioread
        import numpy as np

        logger.debug("Using audioread to load audio file")
        try:
            with audioread.audio_open(path) as audio_file:
                sr = audio_file.samplerate
                logger.debug(f"Original sample rate: {sr}")

                # Read audio data
                audio_data = []
                for block in audio_file:
                    audio_data.append(np.frombuffer(block, dtype=np.int16))

                audio_np = np.concatenate(audio_data).astype(np.float32) / 32768.0  # Convert to float32 and normalize

                # Convert stereo to mono if needed
                if len(audio_np.shape) > 1 and audio_np.shape[1] > 1:
                    logger.debug(f"Converting stereo to mono, shape before: {audio_np.shape}")
                    audio_np = np.mean(audio_np, axis=1)

                logger.debug(f"Audio loaded, shape: {audio_np.shape}, sr: {sr}")

                # Resample if needed
                if sr != target_sr:
                    logger.debug(f"Resampling from {sr} to {target_sr}")
                    import librosa
                    audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=target_sr)

                # Apply processing
                import librosa
                audio_np = trim_silence(audio_np, target_sr)
                audio_np = rms_normalize(audio_np)

                logger.debug(f"Audio preprocessing complete, shape: {audio_np.shape}")
                return torch.tensor(audio_np, dtype=torch.float32)

        except audioread.exceptions.NoBackendError:
            logger.error("No audioread backend available. Falling back to librosa.load")
            import librosa
            audio_np, sr = librosa.load(path, sr=None, mono=True)
            logger.debug(f"Audio loaded with librosa, shape: {audio_np.shape}, sr: {sr}")

            if sr != target_sr:
                logger.debug(f"Resampling from {sr} to {target_sr}")
                audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=target_sr)

            audio_np = trim_silence(audio_np, target_sr)
            audio_np = rms_normalize(audio_np)

            logger.debug(f"Audio preprocessing complete, shape: {audio_np.shape}")
            return torch.tensor(audio_np, dtype=torch.float32)

    except Exception as e:
        logger.error(f"Error in preprocess_audio: {str(e)}")
        logger.error(traceback.format_exc())
        raise


# === DEPENDENCY ===
def get_model_manager():
    """Dependency to ensure model is initialized before handling requests"""
    if not model_manager.is_initialized:
        logger.info("Model not initialized, initializing now...")
        if not model_manager.initialize():
            logger.error("Failed to initialize model")
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
    logger.info("🚀 Starting FastAPI server...")
    # Use background task to initialize model asynchronously
    background_tasks = BackgroundTasks()
    background_tasks.add_task(model_manager.initialize)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for the app"""
    error_msg = str(exc)
    error_type = type(exc).__name__
    error_trace = traceback.format_exc()

    logger.error(f"Unhandled exception: {error_msg}")
    logger.error(error_trace)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            details=error_msg,
            error_type=error_type,
            error_location=str(request.url)
        ).dict()
    )


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
    logger.info(f"Received pronunciation check request for letter: {letter}")

    # Validate letter
    if letter not in model_manager.reference_embeddings:
        logger.warning(f"No reference embeddings for letter '{letter}'")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No reference embeddings for letter '{letter}'"
        )

    if letter not in THRESHOLDS:
        logger.warning(f"No threshold defined for letter '{letter}'")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No threshold defined for letter '{letter}'"
        )

    # Process audio
    temp_path = None
    try:
        # Log audio file info
        logger.info(f"Processing audio file: {audio.filename}, content-type: {audio.content_type}")

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            contents = await audio.read()
            logger.debug(f"Audio file size: {len(contents)} bytes")
            temp_file.write(contents)
            temp_path = temp_file.name
            logger.debug(f"Saved to temporary file: {temp_path}")

        # Process the audio file
        logger.info("Preprocessing audio...")
        input_audio = preprocess_audio(temp_path)

        logger.info("Generating embedding...")
        input_emb = model_manager.get_embedding(input_audio)

        # Calculate similarity with all reference embeddings for this letter
        logger.info(
            f"Calculating similarity with {len(model_manager.reference_embeddings[letter])} reference embeddings")
        similarities = [1 - cosine(ref_emb, input_emb) for ref_emb in model_manager.reference_embeddings[letter]]
        avg_similarity = float(np.mean(similarities))
        threshold = THRESHOLDS[letter]
        match = avg_similarity > threshold

        logger.info(f"Similarity: {avg_similarity}, Threshold: {threshold}, Match: {match}")

        # Return result
        return PronunciationResponse(
            letter=letter,
            similarity=avg_similarity,
            threshold=threshold,
            correct=match
        )

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Error processing audio for letter '{letter}': {str(e)}")
        logger.error(error_trace)

        # Provide more detailed error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": f"Error processing audio: {str(e)}",
                "error_type": type(e).__name__,
                "details": str(e)
            }
        )
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                logger.debug(f"Cleaning up temporary file: {temp_path}")
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": ErrorResponse}}
)
async def health_check(model_manager: ModelManager = Depends(get_model_manager)):
    """Check if the API is ready to process requests"""
    logger.info("Health check requested")
    return HealthResponse(
        status="Ready",
        letters_available=list(model_manager.reference_embeddings.keys()),
        device=str(model_manager.device),
        embeddings_count={letter: len(embs) for letter, embs in model_manager.reference_embeddings.items()}
    )


if __name__ == "__main__":
    import uvicorn

    # Start server
