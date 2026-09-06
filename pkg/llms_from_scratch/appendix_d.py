# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from .ch05 import calc_loss_batch, evaluate_model, generate_and_print_sample

import math
import torch


def find_highest_gradient(model):
    max_grad = None
    for param in model.parameters():
        if param.grad is not None:
            grad_values = param.grad.data.flatten()
            max_grad_param = grad_values.max()
            if max_grad is None or max_grad_param > max_grad:
                max_grad = max_grad_param
    return max_grad


def train_model(model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_freq, eval_iter, start_context, tokenizer,
                warmup_steps, initial_lr=3e-05, min_lr=1e-6, orig_book_version=False):

    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1

    # Optimize ediciden maksimum öğrenme oranını al
    peak_lr = optimizer.param_groups[0]["lr"]

    # Eğitim sürecindeki toplam yineleme sayısını hesapla
    total_training_steps = len(train_loader) * n_epochs

    # Isınma aşamasındaki öğrenme oranı artışını hesapla
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # Öğrenme oranını mevcut aşamaya göre ayarla (ısınma veya kosinüs tavlaması)
            if global_step < warmup_steps:
                # Doğrusal ısınma
                lr = initial_lr + global_step * lr_increment
            else:
                # Isınmadan sonra kosinüs tavlaması
                progress = ((global_step - warmup_steps) /
                            (total_training_steps - warmup_steps))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            # Hesaplanan öğrenme oranını optimize ediciye uygula
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)  # Geçerli öğrenme oranını sakla

            # Kaybı hesapla ve geri yay
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            # Patlayan gradyanlardan kaçınmak için ısınma aşamasından sonra gradyan kırpma uygula
            if orig_book_version:
                if global_step > warmup_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            else:
                if global_step >= warmup_steps:  # kitapta özgün olarak global_step > warmup_steps kullanılmıştı; bu, ısınmadan sonra bir kırpma adımının atlanmasına yol açıyordu
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            tokens_seen += input_batch.numel()

            # Modeli eğitim ve doğrulama kümelerinde belirli aralıklarla değerlendir
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader,
                    device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                # Mevcut kayıpları yazdır
                print(f"Ep {epoch+1} (Iter {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")

        # İlerlemeyi izlemek için modelden bir örnek üret ve yazdır
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen, track_lrs
