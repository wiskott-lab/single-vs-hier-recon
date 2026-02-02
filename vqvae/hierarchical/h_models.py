import torch
from torch import nn
from torch.nn import functional as F

import distributed as dist_fn
import numpy as np
from vector_quantize_pytorch import VectorQuantize

# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )
        
        elif stride == 6:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(channel // 2, channel // 4, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 4, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class UnconditionedHVQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        codebook_dim=64,
        n_embed=512,
        decay=0.99,
        rotation_trick=False,
        kmeans_init=False,
        learnable_codebook = False,
        ema_update = True,
        threshold_ema_dead_code = 2
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_conv_b = nn.Conv2d(channel, embed_dim, 1)
        # self.quantize_t = Quantize(embed_dim, 512)
        self.quantize_t = VectorQuantize(dim=embed_dim, codebook_dim=codebook_dim, codebook_size=n_embed, decay=decay, rotation_trick= rotation_trick, kmeans_init = kmeans_init, learnable_codebook=learnable_codebook, ema_update=ema_update, threshold_ema_dead_code=threshold_ema_dead_code)
        self.quantize_b = VectorQuantize(dim=embed_dim, codebook_dim=codebook_dim, codebook_size=n_embed, decay=decay, rotation_trick= rotation_trick, kmeans_init = kmeans_init, learnable_codebook=learnable_codebook, ema_update=ema_update, threshold_ema_dead_code=threshold_ema_dead_code)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )
        self.vocab_size = n_embed
    def forward(self, input):
        quant_t, quant_b, diff, indices_t, indices_b = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff, indices_b, indices_t

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)
        enc_t = self.quantize_conv_t(enc_t)
        B_t, C_t, H_t, W_t = enc_t.shape
        quant_t = enc_t.permute(0, 2, 3, 1).reshape(B_t, H_t * W_t, C_t)
        quant_t, id_t, diff_t = self.quantize_t(quant_t)
        quant_t = quant_t.view(B_t, H_t, W_t, C_t).permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        quant_b = self.quantize_conv_b(enc_b)
        B, C, H, W = quant_b.shape
        quant_b = quant_b.permute(0, 2, 3, 1).reshape(B, H * W, C)
        quant_b, id_b, diff_b = self.quantize_b(quant_b)
        quant_b = quant_b.view(B, H, W, C).permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        #print(f'code_t: {code_t}')
        quant_t = self.quantize_t.embed_code(code_t)
        #print(f'quantize_t: {quant_t.size()}')
        quant_t = quant_t.permute(0, 3, 1, 2)
        #print(f'quantize_t permuted: {quant_t.size()}')
        quant_b = self.quantize_b.embed_code(code_b)
        #print(f'quantize_b: {quant_b.size()}')
        quant_b = quant_b.permute(0, 3, 1, 2)
        #print(f'quantize_b permuted: {quant_b.size()}')

        dec = self.decode(quant_t, quant_b)

        return dec

    def decode_code_zeromask(self, idx, mask_train, top_mask_train, bottom_length, top_length, code_t, code_b):
        top_zero = np.zeros((top_length, top_length, 64), dtype=bool)
        bottom_zero = np.zeros((bottom_length, bottom_length, 64), dtype=bool)
        zero_tensor_top = torch.from_numpy(top_zero)
        zero_tensor_bottom = torch.from_numpy(bottom_zero)

        # for p in range(0,bottom_length*bottom_length):
        #         if(mask_train[idx][p]):
        #             index_b_zeromask[p] = 503
        print(f'code_t: {code_t.size()}')
        quant_t = self.quantize_t.embed_code(code_t)
        print(f'quantize_t: {quant_t.size()}')
        quant_t = quant_t.permute(0, 3, 1, 2)
        #print(f'quantize_t permuted: {quant_t.size()}')
        quant_b = self.quantize_b.embed_code(code_b)
        print(f'quantize_b: {quant_b.size()}')
        quant_b = quant_b.permute(0, 3, 1, 2)
        #print(f'quantize_b permuted: {quant_b.size()}')

        dec = self.decode(quant_t, quant_b)

        return dec

    def get_ecode_bottom_bef_and_after_conv(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b_t = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b_t).permute(0, 2, 3, 1)

        return enc_b, enc_b_t,quant_b
    
