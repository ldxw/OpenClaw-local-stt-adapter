from faster_whisper import WhisperModel

def download_model(model_name):

    WhisperModel(
        model_name,
        device="cpu"
    )
