"""Shared helpers: model loading, layer access, injection/capture hooks, prompts."""

import json
from contextlib import contextmanager
from pathlib import Path

import torch

MODEL_ID = "google/gemma-3-27b-it"
INJECT_LAYER = 37  # output of decoder layer 37 (Macar et al. default for Gemma3-27B)
READ_LAYER = 43    # downstream read-out
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# Macar et al. (safety-research/introspection-mechanisms, src/vector_utils.py),
# kept verbatim, including the duplicate "Butterflies".
BASELINE_WORDS = [
    "Desks", "Jackets", "Gondolas", "Laughter", "Intelligence",
    "Bicycles", "Chairs", "Orchestras", "Sand", "Pottery",
    "Arrowheads", "Jewelry", "Daffodils", "Plateaus", "Estuaries",
    "Quilts", "Moments", "Bamboo", "Ravines", "Archives",
    "Hieroglyphs", "Stars", "Clay", "Fossils", "Wildlife",
    "Flour", "Traffic", "Bubbles", "Honey", "Geodes",
    "Magnets", "Ribbons", "Zigzags", "Puzzles", "Tornadoes",
    "Anthills", "Galaxies", "Poverty", "Diamonds", "Universes",
    "Vinegar", "Nebulae", "Knowledge", "Marble", "Fog",
    "Rivers", "Scrolls", "Silhouettes", "Marbles", "Cakes",
    "Valleys", "Whispers", "Pendulums", "Towers", "Tables",
    "Glaciers", "Whirlpools", "Jungles", "Wool", "Anger",
    "Ramparts", "Flowers", "Research", "Hammers", "Clouds",
    "Justice", "Dogs", "Butterflies", "Needles", "Fortresses",
    "Bonfires", "Skyscrapers", "Caravans", "Patience", "Bacon",
    "Velocities", "Smoke", "Electricity", "Sunsets", "Anchors",
    "Parchments", "Courage", "Statues", "Oxygen", "Time",
    "Butterflies", "Fabric", "Pasta", "Snowflakes", "Mountains",
    "Echoes", "Pianos", "Sanctuaries", "Abysses", "Air",
    "Dewdrops", "Gardens", "Literature", "Rice", "Enigmas",
]

# Concrete concepts drawn from the Macar 500-concept list (50 Lindsey + 450 new),
# tagged by category so selection can keep the set diverse.
CANDIDATES = {
    "animal": ["Elephants", "Dolphins", "Spiders", "Penguins", "Octopuses", "Bees", "Kangaroos", "Owls"],
    "food": ["Chocolate", "Coffee", "Cheese", "Lemons", "Sushi", "Garlic", "Bananas"],
    "vehicle": ["Submarines", "Helicopters", "Trains", "Rockets", "Tractors", "Sailboats"],
    "instrument": ["Violins", "Drums", "Bagpipes", "Harps", "Trumpets"],
    "weather": ["Thunderstorms", "Hurricanes", "Rainbows", "Blizzards"],
    "place": ["Canyons", "Islands", "Fjords", "Deserts", "Volcanoes"],
    "plant": ["Roses", "Cacti", "Mushrooms", "Sunflowers"],
    "material": ["Copper", "Glass", "Leather", "Silk"],
    "object": ["Satellites", "Cameras", "Mirrors", "Fountains", "Origami", "Aquariums"],
    "body": ["Teeth", "Skulls"],
    "sport": ["Basketball", "Archery"],
}
# Distractors that are near-orthogonal as vectors but semantically tied to a concept
# (they would compete as an "about" answer), excluded by hand.
DISTRACTOR_EXCLUDE = {
    ("Silk", "Spiders"), ("Fjords", "Sailboats"), ("Cacti", "Canyons"), ("Cacti", "Deserts"),
    ("Islands", "Sailboats"), ("Blizzards", "Deserts"), ("Glass", "Mirrors"),
}
CATEGORY = {c: cat for cat, cs in CANDIDATES.items() for c in cs}
ALL_CANDIDATES = [c for cs in CANDIDATES.values() for c in cs]


def load_model():
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    return model, tok


def get_layers(model):
    return model.model.language_model.layers


def chat(tok, user_msg):
    return tok.apply_chat_template(
        [{"role": "user", "content": user_msg}], tokenize=False, add_generation_prompt=True
    )


def encode(tok, text, device="cuda"):
    # apply_chat_template already emits <bos>
    return tok(text, return_tensors="pt", add_special_tokens=False).to(device)


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _replace(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


@contextmanager
def inject(model, vec, alpha, span=None, layer=INJECT_LAYER):
    """Add alpha*vec to the output of `layer`.

    span=None steers every position (including generated tokens); span=(start, end)
    steers only those prompt positions and nothing during cached decoding.
    """
    def hook(mod, inp, out):
        h = _hidden(out)
        add = (alpha * vec).to(h.device, h.dtype)
        if span is None:
            return _replace(out, h + add)
        s, e = span
        if s >= h.shape[1]:
            return out
        h = h.clone()
        h[:, s:e] += add
        return _replace(out, h)

    handle = get_layers(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def capture(model, layers):
    """Record the (post-hook) output hidden states of each layer in `layers`."""
    store, handles = {}, []
    for li in layers:
        def hook(mod, inp, out, li=li):
            store[li] = _hidden(out).detach()
        handles.append(get_layers(model)[li].register_forward_hook(hook))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def load_json(path):
    return json.loads(Path(path).read_text())


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
