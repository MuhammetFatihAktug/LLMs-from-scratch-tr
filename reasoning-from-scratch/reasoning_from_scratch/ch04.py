# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

from .ch02 import generate_text_basic_stream_cache
from .ch03 import extract_final_candidate
from .qwen3 import KVCache

from collections import Counter
import torch


def generate_text_stream_concat_flex(
    model, tokenizer, prompt, device, max_new_tokens,
    verbose=False,
    generate_func=None,  # Yeni
    **generate_kwargs  # Yeni
):

    if generate_func is None:  # Yeni
        generate_func = generate_text_basic_stream_cache

    input_ids = torch.tensor(
        tokenizer.encode(prompt), device=device
        ).unsqueeze(0)

    generated_ids = []
    for token in generate_func(  # Yeni
        model=model,
        token_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        **generate_kwargs,  # Yeni
    ):
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id.item())

        if verbose:
            print(
                tokenizer.decode(next_token_id.tolist()),
                end="",
                flush=True
            )
    return tokenizer.decode(generated_ids)


def plot_scores_bar(
    next_token_logits, start=19_800, end=19_900,
    arrow=True, ylabel="Logit value"
):

    import matplotlib.pyplot as plt

    # Sözcük dağarcığı alt bölümünü seç
    x = torch.arange(start, end)

    # .cpu() is a shortcut for to(torch.device("cpu"))
    logits_section = next_token_logits[0, start:end].float().cpu()

    # Logit'leri çizdir
    plt.bar(x, logits_section)
    plt.xlabel("Vocabulary index")
    plt.ylabel(ylabel)

    # En yüksek logit'i vurgula
    if arrow:
        max_idx = torch.argmax(logits_section)
        plt.annotate(
            "Berlin",
            xy=(x[max_idx], logits_section[max_idx]),
            xytext=(x[max_idx] - 25, logits_section[max_idx] - 2),
            arrowprops={
                "facecolor": "black", "arrowstyle": "->", "lw": 1.5
            },
            fontsize=10,
        )

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def scale_logits_by_temperature(logits, temperature):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return logits / temperature


def plot_logits_with_temperature(
    next_token_logits, start=19_800, end=19_900,
    temps=(0.5, 5.0),
):

    import matplotlib.pyplot as plt

    x = torch.arange(start, end)
    logits_orig = next_token_logits[0, start:end].float().cpu()

    # Sıcaklık ölçeklemesini uygula
    logits_scaled = [
        scale_logits_by_temperature(logits_orig, T) for T in temps
    ]
    # Logit'leri çizdir
    plt.plot(x, logits_orig, label="Original logits", lw=2)
    plt.plot(
        x, logits_scaled[0],
        label=f"T={temps[0]} (sharper)", ls="--", lw=1
    )
    plt.plot(
        x, logits_scaled[1],
        label=f"T={temps[1]} (flatter)", ls=":", lw=3
    )

    # En yüksek logit'i vurgula
    max_idx = torch.argmax(logits_orig)
    plt.annotate(
        "Berlin",
        xy=(x[max_idx], logits_orig[max_idx]),
        xytext=(x[max_idx] - 25, logits_orig[max_idx] + 2),
        arrowprops={"facecolor": "black", "arrowstyle": "->", "lw": 1.5},
        fontsize=12,
    )

    plt.xlabel("Vocabulary index")
    plt.ylabel("Logit value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def count_samples(probas, num_samples=1000, threshold=1, tokenizer=None):
    # Olasılıklara göre örnekler çek
    samples = torch.multinomial(
        probas.cpu(), num_samples=num_samples, replacement=True
    )

    # Her dizinin kaç kez seçildiğini say
    counts = torch.bincount(samples.squeeze(0), minlength=1)

    # Sonuçları yazdır
    for i, c in enumerate(counts):
        if c > threshold:
            if tokenizer is None:
                print(f"Vocab index {i}: {c.item()}x")
            else:
                print(f"'{tokenizer.decode([i])}': {c.item()}x")


@torch.inference_mode()
def generate_text_temp_stream_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    temperature=0.
):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    # Adım 3.1: Logit'leri al
    out = model(token_ids, cache=cache)[:, -1]
    for _ in range(max_new_tokens):

        ########################################
        # YENİ:
        orig_device = token_ids.device

        if temperature is None or temperature == 0.0:
            next_token = torch.argmax(out, dim=-1, keepdim=True)

        else:
            # Adım 3.2: Logit'lere sıcaklık ölçeklemesi uygula
            logits = scale_logits_by_temperature(out, temperature)

            # Adım 3.3: Olasılıklara dönüştür
            probas = torch.softmax(logits, dim=-1)

            # Adım 3.4: Olasılıklara göre token örnekle
            next_token = torch.multinomial(probas.cpu(), num_samples=1)
            next_token = next_token.to(orig_device)

        #########################################
        if (eos_token_id is not None
                and torch.all(next_token == eos_token_id)):
            break

        yield next_token
        out = model(next_token, cache=cache)[:, -1]


def top_p_filter(probas, top_p):
    if top_p is None or top_p >= 1.0:
        return probas

    # Adım 4.1: Olasılığa göre azalan sırada sırala
    sorted_probas, sorted_idx = torch.sort(probas, dim=1, descending=True)

    # Adım 4.2: Kümülatif toplam
    cumprobas = torch.cumsum(sorted_probas, dim=1)

    # Adım 4.3.1: Önek kümülatif kütlesi (token'dan önce) < top_ps olan token'ları tut
    # Example: [0.5, 0.41, 0.09] with top_p=0.9 should keep the first two tokens
    prefix = cumprobas - sorted_probas   # her token'dan önceki kümülatif kütle
    keep = prefix < top_p
    # Her zaman en az bir token tut (çok küçük/pozitif olmayan top_p için yedek)
    keep[:, 0] = True

    # Adım 4.3.2: Kesme noktasının ötesini sıfırla
    kept_sorted = torch.where(
        keep, sorted_probas,
        torch.zeros_like(sorted_probas)
    )
    # Adım 4.3.3: Özgün sıraya geri eşle
    filtered = torch.zeros_like(probas).scatter(1, sorted_idx, kept_sorted)

    # Adım 4.4: Toplamı 1 olacak şekilde yeniden normalleştir
    denom = torch.sum(filtered, dim=1, keepdim=True).clamp_min(1e-12)
    # keepdim=True is technically not necessary but it makes the code work in batched cases
    return filtered / denom


@torch.inference_mode()
def generate_text_top_p_stream_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    temperature=0.,
    top_p=None
):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    # Adım 3.1: Logit'leri al
    out = model(token_ids, cache=cache)[:, -1]
    for _ in range(max_new_tokens):

        orig_device = token_ids.device

        if temperature is None or temperature == 0.0:
            next_token = torch.argmax(out, dim=-1, keepdim=True)

        else:
            # Adım 3.2: Logit'lere sıcaklık ölçeklemesi uygula
            logits = scale_logits_by_temperature(out, temperature)

            # Adım 3.3: Olasılıklara dönüştür
            probas = torch.softmax(logits, dim=-1)

            # (Yeni) Adım 4: Olasılıklara top-p süzgeci uygula
            probas = top_p_filter(probas, top_p)

            # Adım 3.4: Olasılıklara göre token örnekle
            next_token = torch.multinomial(probas.cpu(), num_samples=1)
            next_token = next_token.to(orig_device)

        if (eos_token_id is not None
                and torch.all(next_token == eos_token_id)):
            break

        yield next_token
        out = model(next_token, cache=cache)[:, -1]


def self_consistency_vote(
    model, tokenizer, prompt, device,
    num_samples=10, temperature=0.8, top_p=0.9, max_new_tokens=2048,
    show_progress=True, show_long_answer=False, seed=None,
):
    full_answers, short_answers = [], []

    # 1) Birden çok yanıt örnekle
    for i in range(num_samples):
        if seed is not None:
            torch.manual_seed(seed + i + 1)

        answer = generate_text_stream_concat_flex(
            model=model, tokenizer=tokenizer, prompt=prompt, device=device,
            max_new_tokens=max_new_tokens, verbose=show_long_answer,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature, top_p=top_p,
        )

        # 2) Her yanıttan nihai (kısa) yanıtı ayıkla
        short = extract_final_candidate(
            answer, fallback="number_then_full"
        )
        full_answers.append(answer)
        short_answers.append(short)
        if show_progress:
            print(f"[Sample {i+1}/{num_samples}] → {short!r}")

    # 3) En sık geçen nihai yanıtı seç (öz tutarlılık oylaması)
    counts = Counter(short_answers)
    groups = {s: [] for s in counts}
    for idx, s in enumerate(short_answers):
        groups[s].append(idx)

    mc = counts.most_common()
    if not mc:
        majority_winners, final_answer = [], None
    else:
        top_freq = mc[0][1]
        majority_winners = [s for s, f in mc if f == top_freq]
        final_answer = mc[0][0] if len(majority_winners) == 1 else None

    return {
        "full_answers": full_answers,
        "short_answers": short_answers,
        "counts": dict(counts),
        "groups": groups,
        "majority_winners": majority_winners,
        "final_answer": final_answer,
    }
