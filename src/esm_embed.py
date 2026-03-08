import torch
import esm


_device = "cuda" if torch.cuda.is_available() else "cpu"

_model_cache = {}
_alphabet_cache = {}
_batch_converter_cache = {}


def _load_model(model_name):
    if model_name not in _model_cache:
        model, alphabet = getattr(esm.pretrained, model_name)()
        model = model.to(_device)
        model.eval()

        _model_cache[model_name] = model
        _alphabet_cache[model_name] = alphabet
        _batch_converter_cache[model_name] = alphabet.get_batch_converter()


def get_esm2_residue_embeddings(sequence, model_name="esm2_t6_8M_UR50D"):
    """
    Args:
        sequence: str
        model_name: name from esm.pretrained

    Returns:
        torch.Tensor [L, d]
    """

    _load_model(model_name)

    model = _model_cache[model_name]
    batch_converter = _batch_converter_cache[model_name]

    data = [("protein", sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(_device)

    with torch.no_grad():
        out = model(tokens, repr_layers=[6], return_contacts=False)

    reps = out["representations"][6]  # [1, L+2, d]

    # remove BOS and EOS
    reps = reps[0, 1:-1]

    return reps.cpu()


class ESMEmbedder:
    def __init__(self, model_name="esm2_t6_8M_UR50D", device=None):
        global _device
        if device is not None:
            _device = device
        self.model_name = model_name

    def embed_sequence(self, sequence: str) -> torch.Tensor:
        return get_esm2_residue_embeddings(sequence, model_name=self.model_name)
