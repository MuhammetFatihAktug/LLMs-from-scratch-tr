# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# Bu dosya, şimdiye dek ele aldığımız tüm ilgili kodu
# 2-4. bölümler boyunca bir araya toplar.
# Bu dosya bağımsız bir betik olarak çalıştırılabilir.

import torch


#####################################
# Chapter 5
#####################################
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For döngüsü öncekiyle aynı: logit'leri al ve yalnızca son zaman adımına odaklan
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Yalnızca en yüksek top_k değeri tut
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Yeni (kitapta yok): mps cihazında eşdeğer sonuçlar almak için sayısal kararlılık ipucu
            # softmax'tan önce satır bazında maksimumu çıkar
            logits = logits - logits.max(dim=-1, keepdim=True).values

            # Olasılıkları elde etmek için softmax uygula
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Dağılımdan örnekle
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Aksi hâlde öncekiyle aynı: en yüksek logit değerine sahip sözlük kaydının idx değerini al
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Öncekiyle aynı: örneklenen indeksi süregelen diziye ekle
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx
