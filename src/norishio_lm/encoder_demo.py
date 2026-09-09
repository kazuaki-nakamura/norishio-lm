"""CPU-only forward inspection, not training or evidence of semantic quality."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .encoder import EncoderConfig, MultiChannelEncoder
from .semantic_compiler import SemanticCompiler
from .tensorizer import CHANNELS, SemanticTensorizer


def main() -> None:
    torch.manual_seed(7)
    torch.set_num_threads(1)
    compiler = SemanticCompiler.from_json(Path(__file__).resolve().parents[2] / "data/demo_lexicon.json")
    training_records = [compiler.compile(s) for s in ("性", "性器", "離れない", "離れたくない")]
    tensorizer = SemanticTensorizer.fit(training_records)
    batch = tensorizer.encode(training_records + [compiler.compile("未知語"), compiler.compile("")])
    model = MultiChannelEncoder(EncoderConfig(tensorizer.vocab_sizes)).cpu().eval()
    with torch.no_grad():
        state = model(batch)
        ablated = model(batch, ablation={"etymology": False, "subcharacters": False})
        empty = model(batch, ablation={name: False for name in CHANNELS})
    print(json.dumps({
        "notice": "Randomly initialized CPU forward; hand-authored dictionary; no training or quality claim.",
        "seed": 7, "torch": torch.__version__, "device": str(state.fused.device),
        "channels": list(CHANNELS),
        "input_shapes": {name: list(c.ids.shape) for name, c in batch.channels.items()},
        "fused_shape": list(state.fused.shape), "gate_shape": list(state.gates.shape),
        "finite": bool(torch.isfinite(state.fused).all()),
        "gate_sums": state.gates.sum(dim=1).tolist(),
        "glyph_etymology_disabled": bool((ablated.gates[:, [CHANNELS.index("subcharacters"),
                                                            CHANNELS.index("etymology")]] == 0).all()),
        "all_disabled_zero": bool((empty.fused == 0).all()),
    }, indent=2))


if __name__ == "__main__":
    main()
