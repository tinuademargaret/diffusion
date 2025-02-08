import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from .nn import conv_nd, normalization, linear, zero_module

"""
Unet model architecture

nn.Module features
    Normalization
    Conv

Building blocks 
    time embed
    Timeblocks
    Resblock
        in_layer
        emb_layer
        out_layer
        skip_connection
    Attnblock
        normalization
        convolution
        actual_attention
        conv
    Downsample
    Upsample

Unet Class
    time embed layer

    #Input blocks

    #Middle blocks

    #Output blocks

    # Forward pass
        # create sinusodial embeddings to convert scalar timesteps to vectors (temporal information)
        # Add temporal information to the image feature
"""


class TimeBlock(nn.Module):

    @abstractmethod
    def forward(self, x, emb):
        """"""


class TimeEmbedSequential(nn.Sequential, TimeBlock):

    def forward(self, x, emb):
        # basically for layer in nn.Sequential, nn.Sequential is also a nn.Module type
        for layer in self:
            if isinstance(layer, TimeBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class ResBlock(TimeBlock):
    """
    residual path -> h=in_layers(x)→emb_layers(emb)→out_layers(h)
    in_layer - norm -> silu -> conv ->
    emb_layer - silu -> linear
    out_layer - norm -> silu -> dropout -> conv -> zero_module

    skip_connection -> conv(x) + h
    """

    def __init__(
        self,
        channels,
        emb_dim,
        dims,
        out_channels,
        use_scale_shift_norm,
        dropout,
        use_conv,
        use_checkpoint,
    ):
        super().__init__()
        self.channels = channels
        self.emb_dim = emb_dim
        self.out_channels = out_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.dropout = dropout
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint

        self.in_layer = nn.Sequential(
            normalization(self.channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            linear(
                self.emb_dim,
                (
                    2 * self.out_channels
                    if self.use_scale_shift_norm
                    else self.out_channels
                ),
            ),
        )

        self.out_layer = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU,
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self):
        return

    def _forward(self, x, emb):
        """
        h = x's resid path
        emb_out = emb's resid_path
        resid_path_out = out(h + emmb)

        skip_connection(x) + resid_path_out
        """
        h = self.in_layer(x)
        emb_out = self.emb_layer(emb).type(h.dtype)

        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layer(h)
        return self.skip_conection(x) + h


class AttentionBlock:
    """
    QK, KV circuits

    create  attention pattern i.e a weighing of how much a destination pixel is important to a source pixel
    move the information from a destination token to a source token

    An attention bl

    """

    pass


class UNetModel(nn.Module):
    """
    Each Relu(3x3conv) block = Resblock + Attnblock

    Input blocks
    image -> Relu(3x3conv) -> Relu(3x3conv)
    2x2maxpool
    Relu(3x3conv) -> Relu(3x3conv)
    2x2maxpool
    Relu(3x3conv) -> Relu(3x3conv)
    2x2maxpool
    Relu(3x3conv) -> Relu(3x3conv)
    2x2maxpool
    Relu(3x3conv) -> Relu(3x3conv)

    Middle blocks
    1x1conv

    Outputblocks
    2x2 conv
    Relu(3x3conv) -> Relu(3x3conv)
    2x2 conv
    Relu(3x3conv) -> Relu(3x3conv)
    2x2 conv
    Relu(3x3conv) -> Relu(3x3conv)
    2x2 conv
    Relu(3x3conv) -> Relu(3x3conv)->2x2 conv
    """

    def __init__(self) -> None:
        super().__init()

        """
        Params
            image_size -> 64
            num_channels -> internal channel for unet, img_channel is 3 as per (RGB channels)
            learn_sigma -> whether to learn variation or not
            class-cond -> using classes to condition, we use tempral conditioning
            use_chckpt -> optmization technique (memory for time) not to store all the activations during a fwd pass, but durig bckwd they'll be recomputed
            attn_resolns -> 
            num_heads -> attn layer
            use_scale_shift_norm -> determines how to condition the images with the timestep
            dropout ->
            attention_ds -> when to start including attention layers in unet

        time embed layer(simple mlp)
            linear
            SILU
            linear

        Input Block
            in_blcks = [conv layer] 
            for input level
                for res blocks
                    resblock
                    attn block
                    if not last
                        downsample
                    append to in_blcks
        Middle Block
            resblock
            1attentionblock
            resblock

        Output Block
            for level
                for res blocks
                    reblock
                    attentionblock
                    if not last
                        upsample
                    append to out_blcks
        
        Final Out layer
            normalize
            conv
        """

    def forward(self):
        """
        get emb
        h = InputModuleList(x, emb)
        h = Middleblock(h, emb)
        h = OuputModuleList(h, emb)
        return out(h)
        """
