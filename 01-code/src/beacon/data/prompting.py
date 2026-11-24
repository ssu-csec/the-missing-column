"""Prompt formatting for base vs instruct models.

Base models take the raw 'Instruction: ...\\nQuestion: ...\\nAnswer:' completion prompt and the logit-lens
reads at the 'Answer:' colon. Instruct models must use their CHAT TEMPLATE (special turn tokens); the raw
'Answer:' colon then sits mid-string, so the readout position becomes the LAST token (the generation
position after the assistant marker). These helpers unify both regimes behind a single `chat` flag, so every
model in a comparison is prompted under the same regime.
"""
from __future__ import annotations

import torch

from beacon.models import answer_colon_position
from beacon.data.tokenizer_checks import render_prompt


def _body(prompt: str) -> str:
    """Strip the trailing 'Answer:' so the Instruction+Question becomes the chat user message."""
    return prompt.rsplit("\nAnswer:", 1)[0].strip()


def has_chat_template(loaded) -> bool:
    return bool(getattr(loaded.tokenizer, "chat_template", None))


def _chat_text(loaded, prompt: str) -> str:
    """Render one prompt with the model chat template while disabling Qwen thinking."""
    return render_prompt(loaded.tokenizer, prompt, True)["text"]


def encode_inputs(loaded, prompt: str, chat: bool):
    """Return a device BatchEncoding for one raw or chat-formatted prompt."""
    if chat:
        return loaded.tokenizer(
            _chat_text(loaded, prompt), add_special_tokens=False, return_tensors="pt"
        ).to(loaded.device)
    return loaded.tokenizer(prompt, return_tensors="pt").to(loaded.device)


def encode_batch(loaded, prompts: list[str], chat: bool):
    """Return a left/right-padding-aware BatchEncoding for batched generation."""
    texts = [_chat_text(loaded, p) for p in prompts] if chat else prompts
    return loaded.tokenizer(
        texts,
        add_special_tokens=not chat,
        return_tensors="pt",
        padding=True,
    ).to(loaded.device)


def encode(loaded, prompt: str, chat: bool):
    """Return input_ids [1, T] for the forward. chat=True routes through apply_chat_template (which inserts
    the model's own special/BOS tokens, so we do NOT add them again); chat=False keeps the raw prompt."""
    return encode_inputs(loaded, prompt, chat)["input_ids"]


def readout_col(loaded, prompt: str, chat: bool) -> int:
    """Column index the pivot reads at: the LAST token under a chat template (the model's generation
    position), else the 'Answer:' colon for the raw base prompt."""
    if chat and has_chat_template(loaded):
        return -1
    return answer_colon_position(loaded, prompt)


@torch.no_grad()
def generate_answer(loaded, prompt: str, chat: bool, max_new: int = 12) -> str:
    """Greedy answer text, formatted per base/instruct regime."""
    ids = encode(loaded, prompt, chat)
    out = loaded.model.generate(input_ids=ids, max_new_tokens=max_new, do_sample=False,
                                pad_token_id=loaded.tokenizer.eos_token_id)
    return loaded.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
