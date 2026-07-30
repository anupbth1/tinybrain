import sys
import time
import torch
import importlib.util

# ============================================================
# Repository Path
# ============================================================
REPO = "/content/tinybrain"

sys.path.insert(0, REPO)

# ============================================================
# Load TinyBrain
# ============================================================
spec = importlib.util.spec_from_file_location(
    "tb",
    f"{REPO}/novacore/models/tiny_brain.py"
)

tb = importlib.util.module_from_spec(spec)
sys.modules["tb"] = tb
spec.loader.exec_module(tb)

# ============================================================
# Load Transformer Baseline
# ============================================================
from novacore.core.simple_model import NovaModel
from novacore.core.config import NovaConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("=" * 60)

# ============================================================
# Synthetic Dataset
# ============================================================
torch.manual_seed(0)

SEQ_LEN = 64
VOCAB = 50

data = [torch.randint(2, VOCAB, (SEQ_LEN,)) for _ in range(500)]

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x, x.clone()

train_loader = torch.utils.data.DataLoader(
    Dataset(data[:450]),
    batch_size=32,
    shuffle=True,
)

val_loader = torch.utils.data.DataLoader(
    Dataset(data[450:]),
    batch_size=32,
)

# ============================================================
# Models
# ============================================================

tinybrain = tb.TinyBrainModel(
    tb.TinyBrainConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        num_cells=2,
        memory_slots=4,
        max_think_steps=2,
        output_mlp_hidden=64,
    )
).to(device)

transformer = NovaModel(
    NovaConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        num_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_seq_length=SEQ_LEN,
    )
).to(device)

MODELS = [
    ("TinyBrain", tinybrain),
    ("Transformer", transformer),
]

# ============================================================
# Training
# ============================================================

STEPS = 100

for name, model in MODELS:

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    model.train()

    start = time.time()

    for step in range(STEPS):

        losses = []

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            out = model(x, labels=y)

            loss = out["loss"]

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            losses.append(loss.item())

        if (step + 1) % 20 == 0:
            print(
                f"Step {step+1:3d}/{STEPS} | "
                f"Train Loss = {sum(losses)/len(losses):.4f}"
            )

    elapsed = time.time() - start

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_losses = []

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(device)
            y = y.to(device)

            out = model(x, labels=y)

            val_losses.append(out["loss"].item())

    tokens = len(train_loader.dataset) * SEQ_LEN * STEPS

    print("\nRESULT")
    print("-" * 40)
    print(f"Validation Loss : {sum(val_losses)/len(val_losses):.4f}")
    print(f"Training Time   : {elapsed:.2f} sec")
    print(f"Tokens/sec      : {tokens/elapsed:.0f}")

print("\nDone.")