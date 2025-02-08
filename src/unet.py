import torch.nn as nn

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
