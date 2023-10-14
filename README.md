# Pretty
Ai projects


## Encoder

Output of the Variational Autoencoder (VAE) is the mean and the variance of the latent space.

### Why using these special modules in the encoder, why theses changes in size and channels ?

That's because they work in practice, there is no special reason for that, stable diffusion creators just saw it was working better.
Most encoders work like this, you reduce the size of the image and increase the number of channels, it's a common practice in image processing. (Number of pixels becomes smaller, but the number of information per pixel increases)
