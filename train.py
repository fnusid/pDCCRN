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
# from dc_crn import DCCRN 
# from ConvTasNet import ConvTasNet
import torchaudio
from torchaudio.models import ConvTasNet
from loss import PITSiSNRLoss

from metrics import SE_metrics
import wandb
import sys


import random
random.seed(42)
import warnings
warnings.filterwarnings("ignore")





class E2EpSE(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        finetune_encoder: bool = False,
        emb_dim: int = 256,
        speaker_map_path: str = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/3sp/Libri3Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json",
    ):
        super().__init__()
        self.save_hyperparameters()
        with open(speaker_map_path, "r") as f:
            speaker_map = json.load(f)

        device="cuda" if torch.cuda.is_available() else "cpu"   
        #Get the dual-emb model and teacher model
        
        # dual_emb_ckpt_path = "/mnt/disks/data/model_ckpts/librispeech_asp_ft_wavlm_linear_dualemb_tr360/best-epoch=49-val_separation=0.000.ckpt"
        # dual_emb_ckpt = torch.load(dual_emb_ckpt_path, map_location=device)
        # state = strip_dual_model_weights(dual_emb_ckpt["state_dict"])
        # self.dual_emb_model = SpeakerEncoderDualWrapper(emb_dim=emb_dim)
        # self.dual_emb_model.load_state_dict(state, strict=True)
        # self.dual_emb_model.to(device).eval()
        # for param in self.dual_emb_model.parameters():
        #     param.requires_grad = False

        # self.single_sp_model = SingleSpeakerEncoderWrapper(emb_dim=emb_dim)
        # teacher_ckpt_path = "/mnt/disks/data/model_ckpts/librispeech_asp_wavlm_tr360/best-epoch=62-val_separation=0.000.ckpt"
        # ckpt = torch.load(teacher_ckpt_path, map_location="cpu")
        # state = ckpt["state_dict"]

        # filtered = {}
        # for k, v in state.items():
        #     # only keep model.encoder.* or model.wavlm.*, model.projector.*, model.pooling.*
        #     if k.startswith("model.") and ("arcface" not in k):
        #         filtered[k.replace("model.", "", 1)] = v

        # print("Loaded teacher keys:", len(filtered))

        # self.single_sp_model.load_state_dict(filtered, strict=True)
        # self.single_sp_model.eval()
        # for param in self.single_sp_model.parameters():
        #     param.requires_grad = False



        # -----------------------------
        # 3. Embedding metrics (for validation)
        # -----------------------------
        # self.metrics = SE_metrics(device="cpu")  # will overwrite device at runtime
        self.metrics = SE_metrics(fs=16000, device="cpu", use_dnsmos=True)

        # self.model = DCCRN(rnn_units=256,masking_mode='E',use_clstm=True,kernel_num=[32, 64, 128, 256, 256,256])
        self.model = ConvTasNet(num_sources=3)
        self.loss = PITSiSNRLoss()


    def forward(self, wav):
        """
        wav: [B, T] (or [B, 1, T])
        returns: [B, T] or ([B, 1, T])
        """
        return self.model(wav)

    # -----------------------------
    # TRAINING
    # -----------------------------
    def training_step(self, batch, batch_idx):
        mix, source, labels = batch  # mix: [B,T], source: [B,2,T]
        out = self.forward(mix.unsqueeze(1)) #[B,2,T]

        min_len = min(out.shape[-1], source.shape[-1])
        out = out[..., :min_len]
        source = source[..., :min_len]
        loss = self.loss(out, source)

        self.log("train/SI-SNR_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=mix.size(0))
        # self.log("train/assign_margin", margin.mean(), on_step=True, on_epoch=True, prog_bar=False, batch_size=mix.size(0))
        # self.log("train/pick_id_rate", pick_id.float().mean(), on_step=True, on_epoch=True, prog_bar=False, batch_size=mix.size(0))

        return loss

    # -----------------------------
    # VALIDATION (per-batch)
    # -----------------------------


    def validation_step(self, batch, batch_idx):
        mix, source, labels = batch
        out = self.forward(mix.unsqueeze(1)) #[B,2,T]
        min_len = min(out.shape[-1], source.shape[-1])
        out = out[..., :min_len]
        source = source[..., :min_len]
        self.metrics.update(out, source, mix_audio=mix)
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
        mix = mix.to(self.device)          # [B,T]
        src = src.to(self.device)          # [B,2,T]

        # --- helper: SI-SNR (higher is better) ---
        def si_snr(est, ref, eps=1e-8):
            """
            est, ref: [B,T]
            returns:  [B]
            """
            est = est - est.mean(dim=-1, keepdim=True)
            ref = ref - ref.mean(dim=-1, keepdim=True)

            # projection of est on ref
            ref_energy = (ref ** 2).sum(dim=-1, keepdim=True) + eps
            proj = (est * ref).sum(dim=-1, keepdim=True) * ref / ref_energy

            noise = est - proj
            ratio = (proj ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + eps)
            return 10.0 * torch.log10(ratio + eps)

        with torch.no_grad():
            out = self.forward(mix.unsqueeze(1))                 # ConvTasNet: list/tuple of 2 tensors (usually)
            # out = torch.stack(out, dim=1)           # [B,2,T]  (if each out[k] is [B,T])
            pred0_raw = out[:, 0, :]                # [B,T]
            pred1_raw = out[:, 1, :]                # [B,T]
            pred2_raw = out[:, 2, :]                # [B,T]

        tgt0 = src[:, 0, :]                         # [B,T]
        tgt1 = src[:, 1, :]                         # [B,T]
        tgt2 = src[:, 2, :]                         # [B,T]

        # Match lengths (global min so everything aligns for logging)
        min_len = min(mix.shape[-1], tgt0.shape[-1], tgt1.shape[-1], tgt2.shape[-1], pred0_raw.shape[-1], pred1_raw.shape[-1], pred2_raw.shape[-1])
        mix = mix[..., :min_len]
        tgt0 = tgt0[..., :min_len]
        tgt1 = tgt1[..., :min_len]
        tgt2 = tgt2[..., :min_len]
        pred0_raw = pred0_raw[..., :min_len]
        pred1_raw = pred1_raw[..., :min_len]
        pred2_raw = pred2_raw[..., :min_len]

        pred = torch.stack([pred0_raw, pred1_raw, pred2_raw], dim=1)  # [B,3,T]

        # perms: (which pred goes to tgt0, tgt1, tgt2)
        perms = torch.tensor(
            [
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0],
            ],
            device=pred.device,
            dtype=torch.long,
        )  # [6,3]

        # compute PIT scores for each permutation -> [B,6]
        scores = []
        for k in range(perms.size(0)):
            p0, p1, p2 = perms[k].tolist()
            s = si_snr(pred[:, p0, :], tgt0) + si_snr(pred[:, p1, :], tgt1) + si_snr(pred[:, p2, :], tgt2)  # [B]
            scores.append(s)
        scores = torch.stack(scores, dim=1)  # [B,6]

        # best perm per sample
        best_score, best_idx = scores.max(dim=1)  # [B], [B]

        # margin = best - second_best
        top2_vals, _ = torch.topk(scores, k=2, dim=1)  # [B,2]
        second_best = top2_vals[:, 1]
        margin = (top2_vals[:, 0] - top2_vals[:, 1])   # [B]

        # Reorder predictions according to best permutation
        # gather indices for dim=1
        chosen = perms[best_idx]                       # [B,3]
        gather_idx = chosen.unsqueeze(-1).expand(-1, -1, pred.size(-1))  # [B,3,T]
        pred_reordered = torch.gather(pred, dim=1, index=gather_idx)     # [B,3,T]

        pred0 = pred_reordered[:, 0, :]
        pred1 = pred_reordered[:, 1, :]
        pred2 = pred_reordered[:, 2, :]

        run = self.logger.experiment
        sr = getattr(self.trainer.datamodule, "sampling_rate", 16000)

        for i in range(mix.shape[0]):
            m_np  = mix[i].detach().cpu().numpy().astype("float32")

            t0_np = tgt0[i].detach().cpu().numpy().astype("float32")
            t1_np = tgt1[i].detach().cpu().numpy().astype("float32")
            t2_np = tgt2[i].detach().cpu().numpy().astype("float32")

            p0_np = pred0[i].detach().cpu().numpy().astype("float32")
            p1_np = pred1[i].detach().cpu().numpy().astype("float32")
            p2_np = pred2[i].detach().cpu().numpy().astype("float32")

            run.log({f"audio/mix_{i}":  wandb.Audio(m_np,  sample_rate=sr)})

            run.log({f"audio/tgt0_{i}": wandb.Audio(t0_np, sample_rate=sr)})
            run.log({f"audio/pred0_{i}": wandb.Audio(p0_np, sample_rate=sr)})

            run.log({f"audio/tgt1_{i}": wandb.Audio(t1_np, sample_rate=sr)})
            run.log({f"audio/pred1_{i}": wandb.Audio(p1_np, sample_rate=sr)})

            run.log({f"audio/tgt2_{i}": wandb.Audio(t2_np, sample_rate=sr)})
            run.log({f"audio/pred2_{i}": wandb.Audio(p2_np, sample_rate=sr)})

            # log assignment info
            # perm = perms[best_idx[i]].detach().cpu().tolist()  # e.g. [2,0,1]
            # run.log({
            #     f"sel/best_perm_idx_{i}": int(best_idx[i].item()),     # 0..5
            #     f"sel/perm_tgt0_{i}": int(perm[0]),                    # which pred used for tgt0
            #     f"sel/perm_tgt1_{i}": int(perm[1]),
            #     f"sel/perm_tgt2_{i}": int(perm[2]),
            #     f"sel/margin_{i}": float(margin[i].item()),
            #     f"sel/best_score_{i}": float(best_score[i].item()),
            #     f"sel/second_best_{i}": float(second_best[i].item()),
            # })

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
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
    SPEAKER_MAP = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/3sp/Libri3Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json"


    dm = LibriMixDataModule(
        data_root=DATA_ROOT,
        speaker_map_path=SPEAKER_MAP,
        batch_size=2, 
        num_workers=20, # Set this to your preference
        num_speakers=3
    )

    model = E2EpSE(
        lr=1e-4,
        finetune_encoder=False,
        emb_dim=256,
        speaker_map_path=SPEAKER_MAP,   # ONLY train map here
    )

    wandb_logger = WandbLogger(
        project="pDCCRN_3sp",
        name="convtasnet_3sp_sep_",
        # name='test_run',
        log_model=False,
        save_dir="/mnt/disks/data/model_ckpts/convtasnet_3sp_sep_/wandb_logs",
    )

    ckpt = pl.callbacks.ModelCheckpoint(
        monitor="train/SI-SNR_loss",
        mode="min",
        save_top_k=-1,
        filename="best-{epoch}-{val_separation:.3f}",
        dirpath="/mnt/disks/data/model_ckpts/convtasnet_3sp_sep_/"
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
