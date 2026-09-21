"""The method implemented for gpt-oss, end to end.

The four files here are complete copies of ``optscore/model.py``, ``packing.py``, ``scoring.py``
and ``train.py``, so that either implementation can be read on its own. Every line that differs
from the main one is marked ``gpt-oss:``. gpt-oss differs from a plain dense decoder in four ways:

* alternate layers attend only to the last ``sliding_window`` positions, so the packed forward
  builds one mask per layer type, measuring the window on the position ids (which restart after
  the prompt, exactly what a separate pass over prompt + option would see);
* attention sinks, a learned extra logit per head, exist only in the eager attention path, which
  adds the mask to the logits and so takes ``0 / -inf`` floats instead of booleans;
* the experts of the mixture are one fused 3-D parameter per layer rather than ``nn.Linear``
  modules, so LoRA adapts them through peft's ``target_parameters``;
* the weights are stored in 4-bit MXFP4 and are dequantized to bf16 on load.

The scripts pick this implementation for any model whose name contains ``gpt-oss``; run on a
plain dense model it gives the same numbers as the main one (tested).
"""

from optscore.gptoss import model, packing, scoring, train

__all__ = ["model", "packing", "scoring", "train"]
