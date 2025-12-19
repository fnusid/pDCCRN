import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import LibriMixDataModule       
from dc_crn import DCCRN 

from metrics import SE_metrics
import wandb
import sys
sys.path.append("/home/sidharth./codebase/")

from wavlm_single_embedding.model import SpeakerEncoderWrapper as SingleSpeakerEncoderWrapper
from wavlm_dual_embedding.model import SpeakerEncoderDualWrapper 
import random
random.seed(42)
import warnings
warnings.filterwarnings("ignore")

def strip_model_prefix(state):
    new_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            new_state[k[len("model."):]] = v   # remove "model."
        else:
            new_state[k] = v
    return new_state


def strip_dual_model_weights(state):
    new_state = {}
    for k, v in state.items():
        if not k.startswith("model."):
            continue
        k2 = k.replace("model.", "")
        if k2.startswith("single_sp_model.") or k2.startswith("arcface_loss."):
            continue
        new_state[k2] = v
    return new_state

def cosine(a, b):
    """
    a: [B, D]
    b: [B, D]
    returns: [B]
    """
    dot = (a * b).sum(dim=-1)                 # [B]
    an = a.norm(dim=-1) + 1e-8                # [B]
    bn = b.norm(dim=-1) + 1e-8                # [B]
    return dot / (an * bn)


def assign_embeddings_bijective(e1, e2, emb1, emb2):
    """
    e1,e2:   [B,D] mixture-derived (unordered)
    emb1,2:  [B,D] teacher embeddings for source0/source1 (ordered)
    Returns:
      emb_for_0, emb_for_1: [B,D] such that:
        emb_for_0 corresponds to source[:,0] and emb_for_1 to source[:,1]
      pick_id: [B] bool, True means (e1->spk0, e2->spk1), else swapped
      margin:  [B] confidence margin between the two assignments
    """
    c11 = cosine(e1, emb1)  # e1 vs spk0
    c12 = cosine(e1, emb2)  # e1 vs spk1
    c21 = cosine(e2, emb1)  # e2 vs spk0
    c22 = cosine(e2, emb2)  # e2 vs spk1

    score_id   = c11 + c22
    score_swap = c12 + c21

    pick_id = (score_id >= score_swap)          # [B]
    pick_id_u = pick_id.unsqueeze(-1)           # [B,1]

    emb_for_0 = torch.where(pick_id_u, e1, e2)  # spk0 gets e1 if id else e2
    emb_for_1 = torch.where(pick_id_u, e2, e1)  # spk1 gets the other

    margin = (score_id - score_swap).abs()      # [B]
    return emb_for_0, emb_for_1, pick_id, margin

class E2EpSE(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        finetune_encoder: bool = False,
        emb_dim: int = 256,
        speaker_map_path: str = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libriuni_03_08/Libri2Mix_ovl30to80/wav16k/min/metadata/train360_mapping.json",
    ):
        super().__init__()
        self.save_hyperparameters()
        with open(speaker_map_path, "r") as f:
            speaker_map = json.load(f)

        device="cuda" if torch.cuda.is_available() else "cpu"   
        #Get the dual-emb model and teacher model
        
        dual_emb_ckpt_path = "/mnt/disks/data/model_ckpts/librispeech_asp_ft_wavlm_linear_dualemb_tr360/best-epoch=49-val_separation=0.000.ckpt"
        dual_emb_ckpt = torch.load(dual_emb_ckpt_path, map_location=device)
        state = strip_dual_model_weights(dual_emb_ckpt["state_dict"])
        self.dual_emb_model = SpeakerEncoderDualWrapper(emb_dim=emb_dim)
        self.dual_emb_model.load_state_dict(state, strict=True)
        self.dual_emb_model.to(device).eval()
        for param in self.dual_emb_model.parameters():
            param.requires_grad = False

        self.single_sp_model = SingleSpeakerEncoderWrapper(emb_dim=emb_dim)
        teacher_ckpt_path = "/mnt/disks/data/model_ckpts/librispeech_asp_wavlm_tr360/best-epoch=62-val_separation=0.000.ckpt"
        ckpt = torch.load(teacher_ckpt_path, map_location="cpu")
        state = ckpt["state_dict"]

        filtered = {}
        for k, v in state.items():
            # only keep model.encoder.* or model.wavlm.*, model.projector.*, model.pooling.*
            if k.startswith("model.") and ("arcface" not in k):
                filtered[k.replace("model.", "", 1)] = v

        print("Loaded teacher keys:", len(filtered))

        self.single_sp_model.load_state_dict(filtered, strict=True)
        self.single_sp_model.eval()
        for param in self.single_sp_model.parameters():
            param.requires_grad = False



        # -----------------------------
        # 3. Embedding metrics (for validation)
        # -----------------------------
        self.metrics = SE_metrics(device="cpu")  # will overwrite device at runtime

        self.model = DCCRN(rnn_units=256,masking_mode='E',use_clstm=True,kernel_num=[32, 64, 128, 256, 256,256])


    def forward(self, wav, emb=None):
        """
        wav: [B, T] (or [B, 1, T])
        returns: [B, T] or ([B, 1, T])
        """
        return self.model(wav, emb=emb)

    # -----------------------------
    # TRAINING
    # -----------------------------
    def training_step(self, batch, batch_idx):
        mix, source, labels = batch  # mix: [B,T], source: [B,2,T]

        # ---- teacher embeddings from clean sources (oracle) ----
        with torch.no_grad():
            emb1 = self.single_sp_model(source[:, 0, :])  # [B,D]
            emb2 = self.single_sp_model(source[:, 1, :])  # [B,D]

        # ---- dual embeddings from mixture (frozen) ----
        with torch.no_grad():
            embs = self.dual_emb_model(mix)               # [B,2,D]
            e1 = embs[:, 0, :]
            e2 = embs[:, 1, :]

        # ---- bijective assignment: decide which mixture embedding corresponds to which source ----
        emb_for_0, emb_for_1, pick_id, margin = assign_embeddings_bijective(e1, e2, emb1, emb2)

        # ---- two-pass enhancement/separation ----
        y0 = self.forward(mix, emb=emb_for_0)[1]          # [B,T0]
        y1 = self.forward(mix, emb=emb_for_1)[1]          # [B,T1]

        # ---- crop consistently ----
        T = min(y0.shape[-1], y1.shape[-1], source.shape[-1], mix.shape[-1])
        y0 = y0[..., :T]; y1 = y1[..., :T]
        s0 = source[:, 0, :T]
        s1 = source[:, 1, :T]

        # ---- loss (average across the two targets) ----
        loss0 = self.model.loss(y0, s0, loss_mode="SI-SNR")
        loss1 = self.model.loss(y1, s1, loss_mode="SI-SNR")
        loss  = 0.5 * (loss0 + loss1)

        self.log("train/SI-SNR_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=mix.size(0))
        # self.log("train/assign_margin", margin.mean(), on_step=True, on_epoch=True, prog_bar=False, batch_size=mix.size(0))
        # self.log("train/pick_id_rate", pick_id.float().mean(), on_step=True, on_epoch=True, prog_bar=False, batch_size=mix.size(0))

        return loss

    # -----------------------------
    # VALIDATION (per-batch)
    # -----------------------------


    def validation_step(self, batch, batch_idx):
        mix, source, labels = batch

        with torch.no_grad():
            emb1 = self.single_sp_model(source[:, 0, :])
            emb2 = self.single_sp_model(source[:, 1, :])

            embs = self.dual_emb_model(mix)
            e1 = embs[:, 0, :]
            e2 = embs[:, 1, :]

            emb_for_0, emb_for_1, pick_id, margin = assign_embeddings_bijective(e1, e2, emb1, emb2)

            y0 = self.forward(mix, emb=emb_for_0)[1]
            y1 = self.forward(mix, emb=emb_for_1)[1]

        T = min(y0.shape[-1], y1.shape[-1], source.shape[-1], mix.shape[-1])
        y0 = y0[..., :T]; y1 = y1[..., :T]
        s0 = source[:, 0, :T]
        s1 = source[:, 1, :T]

        # update metrics per speaker (counts as 2 items per mixture)
        self.metrics.update(y0, s0)
        self.metrics.update(y1, s1)

        # optional: log assignment stats
        self.log("val/assign_margin", margin.mean(), prog_bar=False, batch_size=mix.size(0))
        self.log("val/pick_id_rate", pick_id.float().mean(), prog_bar=False, batch_size=mix.size(0))

        return {}

    # -----------------------------
    # VALIDATION (end of epoch)
    # -----------------------------
    def on_validation_epoch_end(self):
        # 1) Compute validation metrics
        m = self.metrics.compute()
        for k, v in m.items():
            self.log(f"val/{k}", v, prog_bar=True)
        self.metrics.reset()

        # 2) Keep a fixed batch for consistent qualitative logging
        if not hasattr(self, "fixed_val_batch"):
            mix, src, _ = next(iter(self.trainer.datamodule.val_dataloader()))
            self.fixed_val_batch = (mix[:3].clone(), src[:3].clone())

        mix, src = self.fixed_val_batch
        mix = mix.to(self.device)
        src = src.to(self.device)

        with torch.no_grad():
            # teacher embeddings (clean sources)
            emb0 = self.single_sp_model(src[:, 0, :])  # [B,D]
            emb1 = self.single_sp_model(src[:, 1, :])  # [B,D]

            # mixture embeddings (unordered)
            embs = self.dual_emb_model(mix)            # [B,2,D]
            e1 = embs[:, 0, :]
            e2 = embs[:, 1, :]

            # 2x2 assignment (bijective): choose identity vs swap
            c11 = cosine(e1, emb0)  # e1 vs spk0
            c12 = cosine(e1, emb1)  # e1 vs spk1
            c21 = cosine(e2, emb0)  # e2 vs spk0
            c22 = cosine(e2, emb1)  # e2 vs spk1

            score_id   = c11 + c22
            score_swap = c12 + c21

            pick_id = (score_id >= score_swap)              # [B] bool
            pick_id_u = pick_id.unsqueeze(-1)               # [B,1]
            margin = (score_id - score_swap).abs()          # [B]

            # assigned embeddings per target (bijective)
            emb_for_0 = torch.where(pick_id_u, e1, e2)      # [B,D]
            emb_for_1 = torch.where(pick_id_u, e2, e1)      # [B,D]

            # run enhancement twice
            pred0 = self.forward(mix, emb=emb_for_0)[1]     # [B,T]
            pred1 = self.forward(mix, emb=emb_for_1)[1]     # [B,T]

        # GT targets
        tgt0 = src[:, 0, :]
        tgt1 = src[:, 1, :]

        # Match lengths
        min_len = min(mix.shape[-1], tgt0.shape[-1], tgt1.shape[-1], pred0.shape[-1], pred1.shape[-1])
        mix  = mix[...,  :min_len]
        tgt0 = tgt0[..., :min_len]
        tgt1 = tgt1[..., :min_len]
        pred0 = pred0[..., :min_len]
        pred1 = pred1[..., :min_len]

        run = self.logger.experiment

        # Log each sample
        for i in range(mix.shape[0]):
            m_np  = mix[i].detach().cpu().numpy().astype("float32")

            t0_np = tgt0[i].detach().cpu().numpy().astype("float32")
            p0_np = pred0[i].detach().cpu().numpy().astype("float32")

            t1_np = tgt1[i].detach().cpu().numpy().astype("float32")
            p1_np = pred1[i].detach().cpu().numpy().astype("float32")

            run.log({f"audio/mix_{i}":  wandb.Audio(m_np,  sample_rate=16000)})

            run.log({f"audio/tgt0_{i}": wandb.Audio(t0_np, sample_rate=16000)})
            run.log({f"audio/pred0_{i}": wandb.Audio(p0_np, sample_rate=16000)})

            run.log({f"audio/tgt1_{i}": wandb.Audio(t1_np, sample_rate=16000)})
            run.log({f"audio/pred1_{i}": wandb.Audio(p1_np, sample_rate=16000)})

            # assignment info
            run.log({
                f"sel/pick_id_{i}": int(pick_id[i].item()),      # 1=id (e1->tgt0), 0=swap
                f"sel/margin_{i}": float(margin[i].item()),
            })

    # def get_pred_from_mix(self, mix, source):
    #     """
    #     mix:    [1, T]
    #     source: [1, 2, T]
    #     """
    #     with torch.no_grad():
    #         emb1 = self.single_sp_model(source[:, 0, :])
    #         emb2 = self.single_sp_model(source[:, 1, :])

    #         # randomly choose one target (for personalization)
    #         # but use deterministic behavior in validation:
    #         # emb_tgt = emb1  # always choose source[0] or use both
    #         idx = random.randint(0,1)
    #         if idx==0:
    #             emb_tgt = emb1
    #         else:
    #             emb_tgt = emb2


    #         embs = self.dual_emb_model(mix)
    #         e1 = embs[:, 0, :]
    #         e2 = embs[:, 1, :]

    #         cosine1 = cosine(e1, emb_tgt)
    #         cosine2 = cosine(e2, emb_tgt)
    #         #true
    #         pred_emb = torch.where((cosine1 > cosine2).unsqueeze(-1), e1, e2)
            

    #         pred = self.model(mix, emb=pred_emb)[1]  # [1, T']

    #         # trim
    #         min_len = min(pred.shape[-1], source.shape[-1])
    #         pred = pred[..., :min_len]
    #         tgt = source[:, idx, :min_len]  # or 1 depending on emb_tgt

    #     return pred, tgt

    # -----------------------------
    # OPTIMIZER + SCHEDULER
    # -----------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
        # return optimizer

        # monitor one of the embedding metrics, e.g., separation (higher is better)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min", 
            factor=0.5,
            patience=3,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/SI-SNR_loss",
                "interval": "epoch",
            },
        }


# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":
    DATA_ROOT = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix" 
    SPEAKER_MAP = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json"


    dm = LibriMixDataModule(
        data_root=DATA_ROOT,
        speaker_map_path=SPEAKER_MAP,
        batch_size=8, 
        num_workers=20, # Set this to your preference
        num_speakers=2
    )

    model = E2EpSE(
        lr=1e-4,
        finetune_encoder=False,
        emb_dim=256,
        speaker_map_path=SPEAKER_MAP,   # ONLY train map here
    )

    wandb_logger = WandbLogger(
        project="pDCCRN_2sp",
        name="pDCCRN_2sp_sep",
        # name='test_run',
        log_model=False,
        save_dir="/mnt/disks/data/model_ckpts/pDCCRN_2sp_sep/wandb_logs",
    )

    ckpt = pl.callbacks.ModelCheckpoint(
        monitor="train/SI-SNR_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch}-{val_separation:.3f}",
        dirpath="/mnt/disks/data/model_ckpts/pDCCRN_2sp_sep/"
    )

    trainer = pl.Trainer(
        strategy="ddp",
        accelerator="gpu",
        devices=[0, 1, 2, 3],
        max_epochs=100,
        logger=wandb_logger,
        callbacks=[ckpt],
        gradient_clip_val=5.0,
        enable_checkpointing=True,
        
    )

    # trainer = pl.Trainer(
    #     accelerator='gpu',
    #     devices=[0],
    #     max_epochs=100,
    #     logger=wandb_logger,
    #     overfit_batches=1,
    #     limit_train_batches=1,
    #     limit_val_batches=1,
    #     num_sanity_val_steps=0,
    #     enable_checkpointing=False,
    # )

    # trainer = pl.Trainer(
    #     accelerator="gpu",
    #     devices=1,
    #     max_epochs=1,
    #     limit_train_batches=1,
    #     limit_val_batches=1,
    #     num_sanity_val_steps=0,
    # )
    trainer.fit(model, datamodule=dm)

    # trainer.validate(model, datamodule=dm, ckpt_path = "/mnt/disks/data/model_ckpts/archive_ckpt/pFCCRN_2sp/best-epoch=60-val_separation=0.000.ckpt")
    wandb.finish()
