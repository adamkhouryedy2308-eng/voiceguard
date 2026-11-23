import torch
from transformers import Wav2Vec2Model

from voiceguard.audio import load_clip

MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model():
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.eval()
    return model.to(DEVICE)


def describe(model, path, augment_fn=None):
    audio = load_clip(path)
    if augment_fn is not None:
        audio = augment_fn(audio)
    x = torch.tensor(audio).unsqueeze(0)
    x = (x - x.mean()) / x.std()
    with torch.no_grad():
        output = model(x.to(DEVICE))
    return output.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()