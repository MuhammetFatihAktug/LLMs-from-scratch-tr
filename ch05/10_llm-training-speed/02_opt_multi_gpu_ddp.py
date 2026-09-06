# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


import os
import time

import matplotlib.pyplot as plt
import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken

# YENİ içe aktarmalar (bkz. Ek A):
import platform
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# NEW: function to initialize a distributed process group (1 process / GPU)
# bu, süreçler arası iletişimi sağlar
# (bkz. Ek A):
def ddp_setup(rank, world_size):
    """
    Arguments:
        rank: a unique process ID
        world_size: total number of processes in the group
    """
    # MASTER_ADDR ve MASTER_PORT yalnızca torchrun tarafından tanımlanmamışsa ayarla
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12345"

    # süreç grubunu başlat
    if platform.system() == "Windows":
        # libuv'yi devre dışı bırak; çünkü Windows için PyTorch bu destekle derlenmiyor
        os.environ["USE_LIBUV"] = "0"
        # Windows kullanıcıları arka uç olarak "nccl" yerine "gloo" kullanmak zorunda kalabilir
        # gloo: Facebook Collective Communication Library
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # nccl: NVIDIA Collective Communication Library
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)


#####################################
# Bölüm 2
#####################################


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Metnin tamamını token'lara ayır
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Kitabı max_length uzunluğunda örtüşen dizilere bölmek için kayan pencere kullan
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# NEW: Modify to set shuffle=False and use a sampler
# (Bkz. Ek A):
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, drop_last=True, num_workers=0):
    # Tokenizer'ı başlat
    tokenizer = tiktoken.get_encoding("gpt2")

    # Veri kümesini oluştur
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Veri yükleyiciyi oluştur
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,  # NEW: False because of DistributedSampler below
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        # NEW: chunk batches across GPUs without overlapping samples:
        sampler=DistributedSampler(dataset)  # YENİ
    )
    return dataloader


#####################################
# Bölüm 3
#####################################
class PyTorchMultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "d_out is indivisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)

        # Başları birleştir; burada self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        context_vec = self.proj(context_vec)

        return context_vec


#####################################
# Bölüm 4
#####################################


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = PyTorchMultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Dikkat bloğu için kestirme (shortcut) bağlantı
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Şekil [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Özgün girdiyi geri ekle

        # İleri beslemeli blok için kestirme (shortcut) bağlantı
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Özgün girdiyi geri ekle

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Şekil [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx, mevcut bağlamdaki indekslerin (B, T) boyutlu dizisidir
    for _ in range(max_new_tokens):

        # Desteklenen bağlam boyutunu aşıyorsa mevcut bağlamı kırp
        # Ör. LLM yalnızca 5 token destekliyorsa ve bağlam boyutu 10 ise
        # bağlam olarak yalnızca son 5 token kullanılır
        idx_cond = idx[:, -context_size:]

        # Tahminleri al
        with torch.no_grad():
            logits = model(idx_cond)

        # Yalnızca son zaman adımına odaklan
        # (batch, n_token, vocab_size) -> (batch, vocab_size) olur
        logits = logits[:, -1, :]

        # En yüksek logit değerine sahip sözlük kaydının idx değerini al
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Örneklenen indeksi süregelen diziye ekle
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

#####################################
# Bölüm 5
#####################################


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # yığın (batch) boyutunu ekle
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # yığın (batch) boyutunu kaldır
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
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


def generate_and_print_sample(model, device, start_context):
    model.eval()

    # NEW: Modify for DDP
    context_size = model.module.pos_emb.weight.shape[0] if isinstance(model, DDP) else model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tiktoken.get_encoding("gpt2")).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tiktoken.get_encoding("gpt2"))
        print(decoded_text.replace("\n", " "))  # Derli toplu yazdırma biçimi
    model.train()


def train_model_simple_with_timing(model, train_loader, val_loader, optimizer, device,
                                   num_epochs, eval_freq, eval_iter, start_context):
    train_losses, val_losses, track_tokens = [], [], []
    total_tokens, global_step, last_tokens = 0, -1, 0

    # NEW: Determine the current rank (default to 0 if not distributed)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    # world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    # Kümülatif ortalama token/sn için değişkenler
    cumulative_tokens, cumulative_time = 0.0, 0.0

    # CUDA'ya özgü zamanlama hazırlığı
    use_cuda = device.type == "cuda"
    if use_cuda:
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()  # Önceki tüm CUDA işlemlerinin bittiğinden emin ol
        t_start.record()          # İlk aralık için zamanlayıcıyı başlat
    else:
        t0 = time.time()          # İlk aralık için zamanlayıcıyı başlat

    # Ana eğitim döngüsü
    for epoch in range(num_epochs):
        # NEW: set epoch for DistributedSampler so each process gets a unique shuffle order
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        for inp_batch, tgt_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # İleri ve geri geçiş
            loss = calc_loss_batch(inp_batch, tgt_batch, model, device)
            loss.backward()
            optimizer.step()

            total_tokens += inp_batch.numel()

            # Değerlendirme aralıklarında geçen süreyi ve saniyedeki token sayısını ölç
            if global_step % eval_freq == 0:
                # Geçerli aralık için zamanlamayı bitir
                if use_cuda:
                    t_end.record()
                    torch.cuda.synchronize()  # Tüm CUDA işlemlerinin tamamlanmasını bekle.
                    elapsed = t_start.elapsed_time(t_end) / 1000  # ms'yi saniyeye çevir
                    t_start.record()  # Zamanlayıcıyı bir sonraki aralık için sıfırla
                else:
                    elapsed = time.time() - t0
                    t0 = time.time()  # Zamanlayıcıyı bir sonraki aralık için sıfırla

                # Bu aralıkta işlenen yerel token sayısını hesapla
                local_interval = total_tokens - last_tokens
                last_tokens = total_tokens

                # Tüm cihazlarda işlenen token sayısını topla
                local_tensor = torch.tensor([local_interval], device=device, dtype=torch.float)
                global_tensor = local_tensor.clone()
                torch.distributed.all_reduce(global_tensor, op=torch.distributed.ReduceOp.SUM)
                global_interval = global_tensor.item()

                # Bu aralık için küresel saniyedeki token sayısı
                global_tps = global_interval / elapsed if elapsed > 0 else 0

                # Kümülatif token sayısını (yerel) güncelle ve küresel olarak topla
                cumulative_tokens += local_interval
                local_cum_tensor = torch.tensor([cumulative_tokens], device=device, dtype=torch.float)
                global_cum_tensor = local_cum_tensor.clone()
                torch.distributed.all_reduce(global_cum_tensor, op=torch.distributed.ReduceOp.SUM)
                global_cumulative_tokens = global_cum_tensor.item()
                cumulative_time += elapsed
                global_avg_tps = global_cumulative_tokens / cumulative_time if cumulative_time > 0 else 0

                # Model başarımını değerlendir (bu ek yük getirebilir)
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens.append(total_tokens)

                # NEW: Only print logs once per GPU (choosing the rank 0 GPU)
                if rank == 0:
                    print(f"Ep {epoch+1}, Step {global_step:06d}, "
                          f"Train: {train_loss:.3f}, Val: {val_loss:.3f}, "
                          f"Step tok/sec: {round(global_tps)}, Global avg tok/sec: {round(global_avg_tps)}")

        # YENİ Üretilen örneği ve bellek kullanım istatistiklerini yalnızca rank 0 yazdırır
        if rank == 0 and epoch % 5 == 0:
            generate_and_print_sample(model, device, start_context)

            # Bellek istatistikleri
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(current_device) / 1024**3  # GB'a çevir
                reserved = torch.cuda.memory_reserved(current_device) / 1024**3    # GB'a çevir

                print(f"\nAllocated memory: {allocated:.4f} GB")
                print(f"Reserved memory: {reserved:.4f} GB\n")

    return train_losses, val_losses, track_tokens


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Eğitim ve doğrulama kaybını dönemlere karşı çiz
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Görülen token'lar için ikinci bir x ekseni oluştur
    ax2 = ax1.twiny()  # Aynı y eksenini paylaşan ikinci bir x ekseni oluştur
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Eksen işaretlerini hizalamak için görünmez çizim
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Yer açmak için yerleşimi ayarla
    # plt.show()


#####################################
# Ana fonksiyon çağrıları
#####################################

# NEW: Add rank and world_size
def main(gpt_config, settings, rank, world_size):

    ddp_setup(rank, world_size)  # NEW: initialize process groups
    device = torch.device("cuda", rank)

    torch.manual_seed(123)

    # NEW: Print info only on 1 GPU
    if rank == 0:
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")

            capability = torch.cuda.get_device_capability()
            if capability[0] >= 7:  # Volta (7.0+), Turing (7.5+), Ampere (8.0+), Hopper (9.0+)
                torch.set_float32_matmul_precision("high")
                print("Uses tensor cores")
            else:
                print("Tensor cores not supported on this GPU. Using default precision.")
        print()

    ##############################
    # Gerekirse veriyi indir
    ##############################

    file_path = "middlemarch.txt"
    url = "https://www.gutenberg.org/cache/epub/145/pg145.txt"

    # NEW: Only download 1 time
    if rank == 0:
        if not os.path.exists(file_path):
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            text_data = response.text
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text_data)
    # NEW: All processes wait until rank 0 is done, using the GPU index.
    torch.distributed.barrier(device_ids=[device.index])

    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    ##############################
    # Modeli başlat
    ##############################

    model = GPTModel(gpt_config)
    model = torch.compile(model)
    model = model.to(device)
    model = model.to(torch.bfloat16)
    # NEW: Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
        fused=True
    )

    ##############################
    # Veri yükleyicileri kur
    ##############################

    # Eğitim/doğrulama oranı
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        num_workers=4
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        num_workers=4
    )

    ##############################
    # Modeli eğit
    ##############################

    train_losses, val_losses, tokens_seen = train_model_simple_with_timing(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=settings["num_epochs"],
        eval_freq=5,
        eval_iter=1,
        start_context="Every effort moves you",
    )

    # NEW: Clean up distributed processes
    destroy_process_group()

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    # NEW: Extract rank and world size from environment variables
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1

    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = 0

    GPT_CONFIG_124M = {
        "vocab_size": 50304,     # Sözcük dağarcığı boyutu
        "context_length": 1024,  # Eğitim örneği başına girdi token'ı
        "emb_dim": 768,          # Gömme (embedding) boyutu
        "n_heads": 12,           # Dikkat başlığı sayısı
        "n_layers": 12,          # Katman sayısı
        "drop_rate": 0.1,        # Dropout oranı
        "qkv_bias": False        # Sorgu-anahtar-değer bias'ı
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,  # * world_size,  # YENİ: Birden çok GPU'yu hesaba katmak için öğrenme oranını artır
        "num_epochs": 50,
        "batch_size": 32,
        "weight_decay": 0.1
    }

    ###########################
    # Eğitimi başlat
    ###########################

    train_losses, val_losses, tokens_seen, model = main(
        GPT_CONFIG_124M, OTHER_SETTINGS,
        rank, world_size  # YENİ
    )

    ###########################
    # Eğitimden sonra
    ###########################

    # NEW: Only create 1 plot
    if rank == 0:
        # Sonuçları çizdir
        epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
        plt.savefig("loss.pdf")

    # Modeli kaydet ve yükle
    #
    # compiled = hasattr(model, "_orig_mod")
    # if compiled:
    #     torch.save(model._orig_mod.state_dict(), "model.pth")
    # else:
    #     torch.save(model.state_dict(), "model.pth")
    #
    # model = GPTModel(GPT_CONFIG_124M)
    # model.load_state_dict(torch.load("model.pth", weights_only=True))
