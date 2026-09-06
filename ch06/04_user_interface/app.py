# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from pathlib import Path
import sys

import tiktoken
import torch
import chainlit

# llms_from_scratch kurulum talimatları için bkz.:
# https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
from llms_from_scratch.ch04 import GPTModel
from llms_from_scratch.ch06 import classify_review


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """
    Code to load finetuned GPT-2 model generated in chapter 6.
    This requires that you run the code in chapter 6 first, which generates the necessary model.pth file.
    """

    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Sözcük dağarcığı boyutu
        "context_length": 1024,  # Bağlam uzunluğu
        "emb_dim": 768,          # Gömme (embedding) boyutu
        "n_heads": 12,           # Dikkat başlığı sayısı
        "n_layers": 12,          # Katman sayısı
        "drop_rate": 0.1,        # Dropout oranı
        "qkv_bias": True         # Sorgu-anahtar-değer bias'ı
    }

    tokenizer = tiktoken.get_encoding("gpt2")

    model_path = Path("..") / "01_main-chapter-code" / "review_classifier.pth"
    if not model_path.exists():
        print(
            f"Could not find the {model_path} file. Please run the chapter 6 code"
            " (ch06.ipynb) to generate the review_classifier.pth file."
        )
        sys.exit()

    # Modeli örnekle
    model = GPTModel(GPT_CONFIG_124M)

    # ch06.ipynb içindeki 6.5 kısmındaki gibi modeli sınıflandırıcıya dönüştür
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=GPT_CONFIG_124M["emb_dim"], out_features=num_classes)

    # Ardından model ağırlıklarını yükle
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return tokenizer, model


# Aşağıdaki chainlit fonksiyonu için gerekli tokenizer ve model dosyalarını edin
tokenizer, model = get_model_and_tokenizer()


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """
    user_input = message.content

    label = classify_review(user_input, model, tokenizer, device, max_length=120)

    await chainlit.Message(
        content=f"{label}",  # Bu, model yanıtını arayüze döndürür
    ).send()
