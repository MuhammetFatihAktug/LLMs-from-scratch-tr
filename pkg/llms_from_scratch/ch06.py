# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


import zipfile
import os
from pathlib import Path

import requests
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
import pandas as pd


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # Dosyayı indirme
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(zip_path, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)

    # Dosyayı açma (unzip)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # .tsv dosya uzantısı ekle
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")


def create_balanced_dataset(df):

    # "spam" örneklerini say
    num_spam = df[df["Label"] == "spam"].shape[0]

    # "spam" örneklerinin sayısına eşit olacak şekilde rastgele "ham" örneği seç
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # "ham" alt kümesini "spam" ile birleştir
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df


def random_split(df, train_frac, validation_frac):
    # DataFrame'in tamamını karıştır
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Ayırma indekslerini hesapla
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # DataFrame'i ayır
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # Metinleri önceden token'lara ayır
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # max_length değerinden uzunlarsa dizileri kırp
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Dizileri en uzun diziye göre doldur
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
        # Note: A more pythonic version to implement this method
        # şudur; bu, bir sonraki bölümde de kullanılıyor:
        # return max(len(encoded_text) for encoded_text in self.encoded_texts)


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # Son çıktı token'ının logit'leri
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Son çıktı token'ının logit'leri
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # num_batches değeri veri yükleyicideki yığın sayısını aşarsa,
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    # Kayıpları ve görülen örnekleri izlemek için listeleri başlat
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Ana eğitim döngüsü
    for epoch in range(num_epochs):
        model.train()  # Modeli eğitim kipine al

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Önceki yığın yinelemesinden kalan kayıp gradyanlarını sıfırla
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Kayıp gradyanlarını hesapla
            optimizer.step()  # Kayıp gradyanlarını kullanarak model ağırlıklarını güncelle
            examples_seen += input_batch.shape[0]  # New: track examples instead of tokens
            global_step += 1

            # İsteğe bağlı değerlendirme adımı
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Her dönemden sonra doğruluğu hesapla
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Eğitim ve doğrulama kaybını dönemlere karşı çiz
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Görülen örnekler için ikinci bir x ekseni oluştur
    ax2 = ax1.twiny()  # Aynı y eksenini paylaşan ikinci bir x ekseni oluştur
    ax2.plot(examples_seen, train_values, alpha=0)  # Eksen işaretlerini hizalamak için görünmez çizim
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Yer açmak için yerleşimi ayarla
    plt.savefig(f"{label}-plot.pdf")
    plt.show()


def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # Model için girdileri hazırla
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    # Note: In the book, this was originally written as pos_emb.weight.shape[1] by mistake
    # Kodu bozmuyordu ama gereksiz kırpmaya yol açardı (1024 yerine 768'e)

    # Çok uzunlarsa dizileri kırp
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # Dizileri en uzun diziye göre doldur
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # yığın (batch) boyutunu ekle

    # Model çıkarımı
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Son çıktı token'ının logit'leri
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Sınıflandırma sonucunu döndür
    return "spam" if predicted_label == 1 else "not spam"
