import torch
from torch import nn
import pytorch_lightning as pl
from vqvae_components import Quantizer, Encoder, Decoder

# Implementation based on https://github.com/rosinality/vq-vae-2-pytorch


class VQVAE(pl.LightningModule):
    def __init__(
            self,
            in_channel=3,
            channel=128,
            n_res_block=2,
            n_res_channel=64,
            embed_dim=64,
            n_embed=512,
            decay=0.99,
            params=None
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantizer(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel
        )
        self.quantize_conv_b = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_b = Quantizer(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel
            # stride=4,
        )

        self.criterion = nn.MSELoss()
        self.latent_loss_weight = 0.25
        if params:
            self.lr = params.lr

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-04)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x = y = train_batch[0]
        x_hat, latent_loss = self.forward(x)
        recon_loss = self.criterion(x_hat, y)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss * self.latent_loss_weight
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = y = val_batch[0]
        x_hat, latent_loss = self.forward(x)
        recon_loss = self.criterion(x_hat, y)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss * self.latent_loss_weight
        self.log('val_loss', loss)

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        # dec_t = self.dec_t(quant_t)
        # enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_latent(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 2, 3, 1)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 2, 3, 1)

        dec = self.decode(quant_t, quant_b)

        return dec


if __name__ == "__main__":
    rand_input = torch.rand((8, 3, 32, 32))

    model = VQVAE(in_channel=3, channel=128, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=512)
    model(rand_input)
