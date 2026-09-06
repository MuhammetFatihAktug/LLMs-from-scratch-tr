# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


import os
from pathlib import Path

import torch
import chainlit

from reasoning_from_scratch.ch02 import (
    get_device,
)
from reasoning_from_scratch.ch02 import generate_text_basic_stream_cache
from reasoning_from_scratch.ch03 import load_model_and_tokenizer, load_tokenizer_only
from reasoning_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B

# ============================================================
# BURAYI DÜZENLEYİN: Basit yapılandırma
# ============================================================
WHICH_MODEL = "reasoning"  # temel model için "base"
MAX_NEW_TOKENS = 38912
LOCAL_DIR = "qwen3"
# LOCAL_DIR içindeki varsayılan ağırlıklar yerine özel bir .pth kontrol noktası
# yüklemek için CHECKPOINT_PATH ayarlayın. WHICH_MODEL'i, o kontrol noktasının
# beklediği tokenizer ile uyumlu tutun; 8. bölüm damıtma kontrol noktaları "reasoning" kullanır.
# Terminal örneği:
#   CHECKPOINT_PATH=/absolute/path/to/model.pth \
#   uv run chainlit run qwen3_chat_interface_multiturn.py
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")
COMPILE = False
# ============================================================


def trim_input_tensor(input_ids_tensor, context_len, max_new_tokens):
    assert max_new_tokens < context_len
    keep_len = max(1, context_len - max_new_tokens)

    # İstem çok uzunsa, keep_len kadar tutmak için baştan kırp
    if input_ids_tensor.shape[1] > keep_len:
        input_ids_tensor = input_ids_tensor[:, -keep_len:]

    return input_ids_tensor


def build_prompt_from_history(history, add_assistant_header=True):
    """
    history: [{"role": "system"|"user"|"assistant", "content": str}, ...]
    """
    parts = []
    for m in history:
        role = m["role"]
        content = m["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

    if add_assistant_header:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


DEVICE = get_device()


def load_app_model_and_tokenizer():
    if CHECKPOINT_PATH is None:
        return load_model_and_tokenizer(
            which_model=WHICH_MODEL,
            device=DEVICE,
            use_compile=COMPILE,
            local_dir=LOCAL_DIR,
        )

    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    tokenizer = load_tokenizer_only(which_model=WHICH_MODEL, local_dir=LOCAL_DIR)
    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.to(DEVICE)

    if COMPILE:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)

    return model, tokenizer


MODEL, TOKENIZER = load_app_model_and_tokenizer()

# Resmî TOKENIZER.eos_token_id ya <|im_end|> (akıl yürütme) ya da
# <|endoftext|> (temel) olsa da, akıl yürütme modeli bazen ikisini birden üretir.
EOS_TOKEN_IDS = (
    TOKENIZER.encode("<|im_end|>")[0],
    TOKENIZER.encode("<|endoftext|>")[0]
)


@chainlit.on_chat_start
async def on_start():
    chainlit.user_session.set("history", [])
    chainlit.user_session.get("history").append(
        {"role": "system", "content": "You are a helpful assistant."}
    )


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """
    # 0) Sohbet geçmişini al ve izle
    history = chainlit.user_session.get("history")
    history.append({"role": "user", "content": message.content})

    # 1) Girdiyi kodla
    prompt = build_prompt_from_history(history, add_assistant_header=True)
    input_ids = TOKENIZER.encode(prompt)
    input_ids_tensor = torch.tensor(input_ids, device=DEVICE).unsqueeze(0)

    # Çok turlu sohbet çok uzayabilir, bu yüzden baştan kırpma ekliyoruz
    input_ids_tensor = trim_input_tensor(
        input_ids_tensor=input_ids_tensor,
        context_len=MODEL.cfg["context_length"],
        max_new_tokens=MAX_NEW_TOKENS
    )

    # 2) İçine akış yapabileceğimiz bir giden mesaj başlat
    out_msg = chainlit.Message(content="")
    await out_msg.send()

    # 3) Akışlı üretim
    for tok in generate_text_basic_stream_cache(
        model=MODEL,
        token_ids=input_ids_tensor,
        max_new_tokens=MAX_NEW_TOKENS,
        # eos_token_id=TOKENIZER.eos_token_id
    ):
        token_id = tok.squeeze(0)
        if token_id in EOS_TOKEN_IDS:
            break
        piece = TOKENIZER.decode(token_id.tolist())
        await out_msg.stream_token(piece)

    # 4) Akıtılan mesajı sonlandır
    await out_msg.update()

    # 5) Sohbet geçmişini güncelle
    history.append({"role": "assistant", "content": out_msg.content})
    chainlit.user_session.set("history", history)
