import math
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from .nn import conv_nd, normalization, linear, zero_module, checkpoint, avg_pool_nd

"""
Unet model architecture

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
    Downsample (downsampling either using convolution or avgpooling)
        if use_conv
            conv(dims, channels, channels, 3,)


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


class Downsample(nn.Module):
    """ """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


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
        dropout,
        dims=2,
        out_channels=None,
        use_scale_shift_norm=False,
        use_conv=False,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_dim = emb_dim
        self.out_channels = (
            out_channels or channels
        )  # to change the number of channels or not
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

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

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


class AttentionBlock(nn.Module):
    """
    reshape x from b,c,w,h -> b,c,(wh) flatten 2d image to 1d
    normalize x and apply transformation to x to get b,3*c,(w*h);to generate different features for q, k, and v simultaneously
    reshape the transformation from b,3*c,(w*h) -> b*num_heads, (c*3) ,(w*h) to create n num_heads each of q, k, v
    compute attention pattern
    reshape back to normal
    add output layer which would also be initially zeroed out
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.num_heads = num_heads
        self.attention = QKVAttention()
        self.use_checkpoint = use_checkpoint
        self.proj_out = nn.Sequential(
            zero_module(conv_nd(1, channels, self.use_checkpoint))
        )

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *w_h = x.shape
        x = x.reshape(
            b, c, -1
        )  # x initallly has 4d shape (batch_size, channels, width, height)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv[2])  # flattens the qkv split
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *w_h)


class QKVAttention(nn.Module):
    """Computes attention pattern"""

    def forward(self, qkv):
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(
            weight.dtype
        )  # converts the weights to a probability distribution
        return torch.einsum("bts,bcs->bct", weight, v)


class UNetModel(nn.Module):
    """
    in_channels: channels in input tensor
    multi_channels: channel multipliers for the model layers
    model_channels: base channel internal model layers
    out_channels: channels in the output tensor
    num_res_block: number residual blocks per downsample
    attention_resolutions: a collection of when attention would be used during downsampling
    dropout: the dropout probability
    conv_resample: use learned convolution for downsampling and upsampling if True
    use_scale_shift_norm: a specific way to combine the image and time embeddings
    use_checkpoint: use checkoint optimization
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        multi_channels=(1, 2, 4, 8),
        dropout=0,
        conv_resample=True,
        dims=2,
        use_scale_shift_norm=False,
        use_checkpoint=False,
        num_heads=1,
    ):
        super().__init()
        self.in_channels = in_channels
        self.multi_channels = multi_channels
        self.num_res_block = num_res_blocks
        self.out_channel = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.conv_resample = conv_resample
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads

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
            dropout ->s
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
        time_embed_dim = model_channels * 4
        # inner model channel size
        self.model_channels = model_channels
        # we'll index this layer with a scalar time step value to get a time vector
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            time_embed_dim,
            time_embed_dim,
        )

        self.input_blocks = nn.ModuleList(
            [
                TimeEmbedSequential(
                    conv_nd(2, self.in_channels, self.model_channels, 3, padding=1)
                )
            ]
        )
        ds = 1
        ch = self.model_channels
        for level, mult in enumerate(self.multi_channels):
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(
                        self.model_channels,
                        time_embed_dim,
                        self.dropout,
                        mult * self.model_channels,
                        use_conv=self.conv_resample,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                ]
            ch *= mult
            if ds in self.attention_resolutions:
                layers.append(
                    AttentionBlock(
                        ch,
                        use_checkpoint=self.use_checkpoint,
                        num_heads=self.num_heads,
                    )
                )
            self.input_blocks.append(TimeEmbedSequential(*layers))
        if level != len(self.multi_channels) - 1:
            self.input_blocks.append(
                TimeEmbedSequential(Downsample(ch, self.conv_resample, dims=dims))
            )

    def forward(self):
        """
        get emb
        h = InputModuleList(x, emb)
        h = Middleblock(h, emb)
        h = OuputModuleList(h, emb)
        return out(h)
        """
