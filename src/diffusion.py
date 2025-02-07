"""
Cifar dataset 128x3x64x64 batch_size is 128

Unet Class
    time embed layer

    #Input blocks

    #Middle blocks

    #Output blocks

    # Forward pass
        # create sinusodial embeddings to convert scalar timesteps to vectors (temporal information)
        # Add temporal information to the image feature

# Algorithm

diffusion class
    get beta from beta scheduler 

Sampler

Dataloader

Train loop #learns the distribution of the data
    # get batch of images and classes if classes
    # run_step
        # forward call
            # sub sample current batch
            # sample t for each sub sample image
            # predict noise for each t and compute loss
                # to compute noise
                    sample guassian noise of the image size
                    sample x_t (noisy image version) using the q_t formula
                    forward pass x_t and t on u_net
                    freeze hybrid loss 
                    compute lt-1 loss
                    compute nlll
                    compute mse between predicted noise and target noise 


Sampling code
    generate guassian noise  
    set timestep 
        for each timestep
            compute model mean 
            get variance and multiply by noise
        at timestep 0 return the image 
"""
