import numpy as np
import torch
from viz.latent_traversals import LatentTraverser
from scipy import stats
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image


class Visualizer():
    def __init__(self, model, pad_value=0.):
        """
        Visualizer is used to generate images of samples, reconstructions,
        latent traversals and so on of the trained model.

        Parameters
        ----------
        model : jointvae.models.VAE instance
        """
        self.model = model
        self.latent_traverser = LatentTraverser(self.model.latent_spec)
        self.save_images = True  # If false, each method returns a tensor
                                 # instead of saving image.
        self.pad_value = pad_value

    def reconstructions(self, data, size=(8, 8), filename='recon.png'):
        """
        Generates reconstructions of data through the model.

        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)

        size : tuple of ints
            Size of grid on which reconstructions will be plotted. The number
            of rows should be even, so that upper half contains true data and
            bottom half contains reconstructions
        """
        # Plot reconstructions in test mode, i.e. without sampling from latent
        self.model.eval()
        # Pass data through VAE to obtain reconstruction
        input_data = Variable(data, volatile=True)
        if self.model.use_cuda:
            input_data = input_data.cuda()
        recon_data, _ = self.model(input_data)
        self.model.train()

        # Upper half of plot will contain data, bottom half will contain
        # reconstructions
        num_images = size[0] * size[1] // 2
        originals = input_data[:num_images].cpu()
        reconstructions = recon_data.view(-1, *self.model.img_size)[:num_images].cpu()
        # If there are fewer examples given than spaces available in grid,
        # augment with blank images
        num_examples = originals.size()[0]
        if num_images > num_examples:
            blank_images = torch.zeros((num_images - num_examples,) + originals.size()[1:])
            originals = torch.cat([originals, blank_images])
            reconstructions = torch.cat([reconstructions, blank_images])

        # Concatenate images and reconstructions
        comparison = torch.cat([originals, reconstructions])

        if self.save_images:
            save_image(comparison.data, filename, nrow=size[0])
        else:
            return make_grid(comparison.data, nrow=size[0], pad_value=self.pad_value)

    def samples(self, size=(8, 8), filename='samples.png'):
        """
        Generates samples from learned distribution by sampling prior and
        decoding.

        size : tuple of ints
        """
        # Get prior samples from latent distribution
        cached_sample_prior = self.latent_traverser.sample_prior
        self.latent_traverser.sample_prior = True
        prior_samples = self.latent_traverser.traverse_grid(size=size)
        self.latent_traverser.sample_prior = cached_sample_prior

        # Map samples through decoder
        generated = self._decode_latents(prior_samples)

        if self.save_images:
            save_image(generated.data, filename, nrow=size[1])
        else:
            return make_grid(generated.data, nrow=size[1], pad_value=self.pad_value)

    def latent_traversal_line(self, cont_idx=None, disc_idx=None, size=8,
                              filename='traversal_line.png'):
        """
        Generates an image traversal through a latent dimension.

        Parameters
        ----------
        See viz.latent_traversals.LatentTraverser.traverse_line for parameter
        documentation.
        """
        # Generate latent traversal
        latent_samples = self.latent_traverser.traverse_line(cont_idx=cont_idx,
                                                             disc_idx=disc_idx,
                                                             size=size)

        # Map samples through decoder
        generated = self._decode_latents(latent_samples)

        if self.save_images:
            save_image(generated.data, filename, nrow=size)
        else:
            return make_grid(generated.data, nrow=size, pad_value=self.pad_value)

    def latent_traversal_grid(self, cont_idx=None, cont_axis=None,
                              disc_idx=None, disc_axis=None, size=(5, 5),
                              filename='traversal_grid.png'):
        """
        Generates a grid of image traversals through two latent dimensions.

        Parameters
        ----------
        See viz.latent_traversals.LatentTraverser.traverse_grid for parameter
        documentation.
        """
        # Generate latent traversal
        latent_samples = self.latent_traverser.traverse_grid(cont_idx=cont_idx,
                                                             cont_axis=cont_axis,
                                                             disc_idx=disc_idx,
                                                             disc_axis=disc_axis,
                                                             size=size)

        # Map samples through decoder
        generated = self._decode_latents(latent_samples)

        if self.save_images:
            save_image(generated.data, filename, nrow=size[1])
        else:
            return make_grid(generated.data, nrow=size[1], pad_value=self.pad_value)

    def all_latent_traversals(self, size=8, filename='all_traversals.png'):
        """
        Traverses all latent dimensions one by one and plots a grid of images
        where each row corresponds to a latent traversal of one latent
        dimension.

        Parameters
        ----------
        size : int
            Number of samples for each latent traversal.
        """
        latent_samples = []

        # Perform line traversal of every continuous and discrete latent
        for cont_idx in range(self.model.latent_cont_dim):
            latent_samples.append(self.latent_traverser.traverse_line(cont_idx=cont_idx,
                                                                      disc_idx=None,
                                                                      size=size))

        for disc_idx in range(self.model.num_disc_latents):
            latent_samples.append(self.latent_traverser.traverse_line(cont_idx=None,
                                                                      disc_idx=disc_idx,
                                                                      size=size))

        # Decode samples
        generated = self._decode_latents(torch.cat(latent_samples, dim=0))

        if self.save_images:
            save_image(generated.data, filename, nrow=size)
        else:
            return make_grid(generated.data, nrow=size, pad_value=self.pad_value)
    
    def latent_traversal_fix_discrete_vary_continous(self, cont_idx=None, disc_idx=None, size=8, discrete_label=0, sample_prior_cont=False,
                                                     filename='traversal_line_fix_discrete_vary_continous.png'):
        """
        Generates an image traversal through a latent dimension.

        Parameters
        ----------
        See viz.latent_traversals.LatentTraverser.traverse_line for parameter
        documentation.
        """
        # Generate latent traversal
        latent_samples = self.latent_traverser.traverse_line_fix_discrete_vary_continous(cont_idx=cont_idx,
                                                             disc_idx=disc_idx,
                                                             discrete_label=discrete_label,
                                                             sample_prior_cont=sample_prior_cont,
                                                             size=size)

        # Map samples through decoder
        generated = self._decode_latents(latent_samples)

        if self.save_images:
            save_image(generated.data, filename, nrow=size)
        else:
            return make_grid(generated.data, nrow=size, pad_value=self.pad_value)
    
    def latent_traversal_mix_discrete_vary_continous(self, cont_idx=None, disc_idx=None, size=8, discrete_labels=[0], weights=[1], sample_prior_cont=False,
                                                     filename='traversal_line_mix_discrete_vary_continous.png'):
        """
        Generates an image traversal through a latent dimension.

        Parameters
        ----------
        See viz.latent_traversals.LatentTraverser.traverse_line for parameter
        documentation.
        """
        # Generate latent traversal
        latent_samples = self.latent_traverser.traverse_line_mix_discrete_vary_continous(cont_idx=cont_idx,
                                                             disc_idx=disc_idx,
                                                             discrete_labels=discrete_labels,
                                                             weights=weights,
                                                             sample_prior_cont=sample_prior_cont,
                                                             size=size)

        # Map samples through decoder
        generated = self._decode_latents(latent_samples)

        if self.save_images:
            save_image(generated.data, filename, nrow=size)
        else:
            return make_grid(generated.data, nrow=size, pad_value=self.pad_value)
    
    def latent_traversal_latent_sample(self, cont_idx=None, latent_sample=None, size=8,
                                        filename='latent_traversal_latent_sample.png'):
        """
        Generates an image traversal through a latent dimension.

        Parameters
        ----------
        See viz.latent_traversals.LatentTraverser.traverse_line for parameter
        documentation.
        """
        # Generate latent traversal
        samples = np.zeros(shape=(size, self.model.latent_cont_dim + self.model.latent_disc_dim))
        cdf_traversal = np.linspace(0.05, 0.95, size)
        cont_traversal = stats.norm.ppf(cdf_traversal)
        for i in range(0,size):
            samples[i] = latent_sample.detach().numpy()
            samples[i][cont_idx] = cdf_traversal[i]
        samples = torch.Tensor(samples)

        # Map samples through decoder
        generated = self._decode_latents(samples)

        if self.save_images:
            save_image(generated.data, filename, nrow=size)
        else:
            return make_grid(generated.data, nrow=size, pad_value=self.pad_value)
    

    def latent_traversal_all_latent_samples_sub_function(self, cont_idx, latent_sample, size=8):
        samples = np.zeros(shape=(size, self.model.latent_cont_dim + self.model.latent_disc_dim))
        cdf_traversal = np.linspace(0.05, 0.95, size)
        cont_traversal = stats.norm.ppf(cdf_traversal)
        for i in range(0,size):
            samples[i] = latent_sample.detach().numpy()
            samples[i][cont_idx] = cdf_traversal[i]
        samples = torch.Tensor(samples)
        return samples
    
    def latent_traversal_all_latent_samples(self, latent_sample=None, size=8,
                                        filename='latent_traversal_all_latent_samples.png'):
        """
        Generates an image traversal through a latent dimension.

        Parameters
        ----------
        See viz.latent_traversals.LatentTraverser.traverse_line for parameter
        documentation.
        """
        # Generate latent traversal
        samples = [self.latent_traversal_all_latent_samples_sub_function(cont_idx=cont_idx, latent_sample=latent_sample, size=size) 
                   for cont_idx in range(self.model.latent_cont_dim)]
        

        # Map samples through decoder
        generated = self._decode_latents(torch.cat(samples, dim=0))

        if self.save_images:
            save_image(generated.data, filename, nrow=size)
        else:
            return make_grid(generated.data, nrow=size, pad_value=self.pad_value)
    
    def latent_traversal_cont_line(self, cont_idx=0,n_samples=8,data=None):
        if data is None:
            # mean of prior for other dimensions
            latent_sample = torch.zeros(n_samples, self.model.latent_cont_dim + self.model.latent_disc_dim)
            cdf_traversal = np.linspace(0.05, 0.95, n_samples)
            cont_traversal = stats.norm.ppf(cdf_traversal)

        else:
            if data.size(0) > 1:
                raise ValueError("Every value should be sampled from the same posterior, but {} datapoints given.".format(data.size(0)))

            with torch.no_grad():
                latent_dist = self.model.encode(data)
                post_mean, post_logvar = latent_dist['cont']
                latent_sample = self.model.reparameterize(latent_dist)
                latent_sample = latent_sample.cpu().repeat(n_samples, 1)
                post_mean_idx = post_mean.detach().numpy()[0, cont_idx]
                post_std_idx = torch.exp(post_logvar / 2).detach().numpy()[0, cont_idx]

            # travers from the gaussian of the posterior in case quantile
            cdf_traversal = np.linspace(0.05, 0.95, n_samples)
            cont_traversal = stats.norm.ppf(cdf_traversal, loc=post_mean_idx, scale=post_std_idx)
            # print(cont_traversal)

        for i in range(n_samples):
            latent_sample[i, cont_idx] = cont_traversal[i]
        # print(latent_sample)
        return latent_sample
    
    def latent_traversal_cont_data(self, data=None, n_latents=None, size=8,
                                        filename='latent_traversal_cont_data.png'):
        """
        Generates an image traversal through a latent dimension.

        Parameters
        ----------
        See viz.latent_traversals.LatentTraverser.traverse_line for parameter
        documentation.
        """
        # Generate latent traversal
        n_latents = n_latents if n_latents is not None else self.model.latent_cont_dim
        latent_samples = [self.latent_traversal_cont_line(cont_idx=dim, n_samples=size, data=data)
                          for dim in range(self.model.latent_cont_dim)]
        generated = self._decode_latents(torch.cat(latent_samples, dim=0))

        if self.save_images:
            save_image(generated.data, filename, nrow=size)
        else:
            return make_grid(generated.data, nrow=size, pad_value=self.pad_value)

    def _decode_latents(self, latent_samples):
        """
        Decodes latent samples into images.

        Parameters
        ----------
        latent_samples : torch.autograd.Variable
            Samples from latent distribution. Shape (N, L) where L is dimension
            of latent distribution.
        """
        latent_samples = Variable(latent_samples)
        if self.model.use_cuda:
            latent_samples = latent_samples.cuda()
        return self.model.decode(latent_samples).cpu()


def reorder_img(orig_img, reorder, by_row=True, img_size=(3, 32, 32), padding=2):
    """
    Reorders rows or columns of an image grid.

    Parameters
    ----------
    orig_img : torch.Tensor
        Original image. Shape (channels, width, height)

    reorder : list of ints
        List corresponding to desired permutation of rows or columns

    by_row : bool
        If True reorders rows, otherwise reorders columns

    img_size : tuple of ints
        Image size following pytorch convention

    padding : int
        Number of pixels used to pad in torchvision.utils.make_grid
    """
    reordered_img = torch.zeros(orig_img.size())
    _, height, width = img_size

    for new_idx, old_idx in enumerate(reorder):
        if by_row:
            start_pix_new = new_idx * (padding + height) + padding
            start_pix_old = old_idx * (padding + height) + padding
            reordered_img[:, start_pix_new:start_pix_new + height, :] = orig_img[:, start_pix_old:start_pix_old + height, :]
        else:
            start_pix_new = new_idx * (padding + width) + padding
            start_pix_old = old_idx * (padding + width) + padding
            reordered_img[:, :, start_pix_new:start_pix_new + width] = orig_img[:, :, start_pix_old:start_pix_old + width]

    return reordered_img