import imageio
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.utils import make_grid
import math

EPS = 1e-12


class Trainer_BTCVAE():
    def __init__(self, model, optimizer, cont_capacity=None,
                 disc_capacity=None, print_loss_every=50, record_loss_every=5,
                 use_cuda=False, is_mss=True):
        """
        Class to handle training of model.

        Parameters
        ----------
        model : jointvae.models.VAE instance

        optimizer : torch.optim.Optimizer instance

        cont_capacity : tuple (float, float, int, float) or None
            Tuple containing (min_capacity, max_capacity, num_iters, gamma_z).
            Parameters to control the capacity of the continuous latent
            channels. Cannot be None if model.is_continuous is True.

        disc_capacity : tuple (float, float, int, float) or None
            Tuple containing (min_capacity, max_capacity, num_iters, gamma_c).
            Parameters to control the capacity of the discrete latent channels.
            Cannot be None if model.is_discrete is True.

        print_loss_every : int
            Frequency with which loss is printed during training.

        record_loss_every : int
            Frequency with which loss is recorded during training.

        use_cuda : bool
            If True moves model and training to GPU.
        """
        self.model = model
        self.optimizer = optimizer
        self.cont_capacity = cont_capacity
        self.disc_capacity = disc_capacity
        self.print_loss_every = print_loss_every
        self.record_loss_every = record_loss_every
        self.use_cuda = use_cuda
        self.is_mss = is_mss
        self.steps_anneal = 10000
        self.alpha_loss = 1
        self.beta = 6
        self.gamma = 1

        if self.model.is_continuous and self.cont_capacity is None:
            raise RuntimeError("Model is continuous but cont_capacity not provided.")

        if self.model.is_discrete and self.disc_capacity is None:
            raise RuntimeError("Model is discrete but disc_capacity not provided.")

        if self.use_cuda:
            self.model.cuda()

        # Initialize attributes
        self.num_steps = 0
        self.batch_size = None
        self.losses = {'loss': [],
                       'recon_loss': [],
                       'kl_loss': []}

        # Keep track of divergence values for each latent variable
        if self.model.is_continuous:
            self.losses['kl_loss_cont'] = []
            # For every dimension of continuous latent variables
            for i in range(self.model.latent_spec['cont']):
                self.losses['kl_loss_cont_' + str(i)] = []

        if self.model.is_discrete:
            self.losses['kl_loss_disc'] = []
            # For every discrete latent variable
            for i in range(len(self.model.latent_spec['disc'])):
                self.losses['kl_loss_disc_' + str(i)] = []

    def train(self, data_loader, epochs=10, save_training_gif=None, save_after_n_epochs=10, resume=False, checkpoint_path='', model_dir='./model_in_progress/'):
        """
        Trains the model.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader

        epochs : int
            Number of epochs to train the model for.

        save_training_gif : None or tuple (string, Visualizer instance)
            If not None, will use visualizer object to create image of samples
            after every epoch and will save gif of these at location specified
            by string. Note that string should end with '.gif'.
        """
        if save_training_gif is not None:
            training_progress_images = []
        
        if resume:
            try:
                checkpoint = torch.load(checkpoint_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except:
                print('Unable to load the checkpoint')
        
        logfile = open(model_dir + 'losses.csv', 'w')

        self.batch_size = data_loader.batch_size
        self.full_data_size = len(data_loader.dataset)
        best_loss = 100000
        self.model.train()
        for epoch in range(epochs):
            mean_epoch_loss = self._train_epoch(data_loader)
            print('Epoch: {} Average loss: {:.2f}'.format(epoch + 1,
                                                          self.batch_size * self.model.num_pixels * mean_epoch_loss))
            line = "{},{:.2f}".format(epoch + 1, self.batch_size * self.model.num_pixels * mean_epoch_loss)
            logfile.write(line)
            logfile.write("\n")

            if save_training_gif is not None:
                # Generate batch of images and convert to grid
                viz = save_training_gif[1]
                viz.save_images = False
                img_grid = viz.all_latent_traversals(size=10)
                # Convert to numpy and transpose axes to fit imageio convention
                # i.e. (width, height, channels)
                img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
                # Add image grid to training progress
                training_progress_images.append(img_grid)
            
            if (epoch + 1) % save_after_n_epochs == 0:
                print('saving checkpoint on epoch: ',epoch+1)
                torch.save(self.model.state_dict(), model_dir + 'model.pt')
                torch.save({
                            'epoch': epoch + 1,
                            'loss': self.batch_size * self.model.num_pixels * mean_epoch_loss,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()
                            },model_dir + 'model-checkpoint.tar')
                if (self.batch_size * self.model.num_pixels * mean_epoch_loss) < best_loss:
                    print('saving best loss checkpoint on epoch: ',epoch+1)
                    torch.save(self.model.state_dict(), model_dir + 'model-best-loss.pt')
                    best_loss = self.batch_size * self.model.num_pixels * mean_epoch_loss
                    torch.save({
                                'epoch': epoch + 1,
                                'loss': self.batch_size * self.model.num_pixels * mean_epoch_loss,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict()
                                },model_dir + 'model-best-loss-checkpoint.tar')

        logfile.close()
        if save_training_gif is not None:
            imageio.mimsave(save_training_gif[0], training_progress_images,
                            fps=24)

    def _train_epoch(self, data_loader):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
        """
        epoch_loss = 0.
        print_every_loss = 0.  # Keeps track of loss to print every
                               # self.print_loss_every
        for batch_idx, (data, label) in enumerate(data_loader):
            iter_loss = self._train_iteration(data)
            epoch_loss += iter_loss
            print_every_loss += iter_loss
            # Print loss info every self.print_loss_every iteration
            if batch_idx % self.print_loss_every == 0:
                if batch_idx == 0:
                    mean_loss = print_every_loss
                else:
                    mean_loss = print_every_loss / self.print_loss_every
                print('{}/{}\tLoss: {:.3f}'.format(batch_idx * len(data),
                                                  len(data_loader.dataset),
                                                  self.model.num_pixels * mean_loss))
                print_every_loss = 0.
        # Return mean epoch loss
        return epoch_loss / len(data_loader.dataset)

    def _train_iteration(self, data):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            A batch of data. Shape (N, C, H, W)
        """
        self.num_steps += 1
        if self.use_cuda:
            data = data.cuda()

        self.optimizer.zero_grad()
        recon_batch, latent_dist = self.model(data)
        loss = self._loss_function(data, recon_batch, latent_dist, self.model.latent_cont_sample)
        loss.backward()
        self.optimizer.step()

        train_loss = loss.item()
        return train_loss

    def _loss_function(self, data, recon_data, latent_dist, latent_cont_sample):
        """
        Calculates loss for a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images). Should have shape (N, C, H, W)

        recon_data : torch.Tensor
            Reconstructed data. Should have shape (N, C, H, W)

        latent_dist : dict
            Dict with keys 'cont' or 'disc' or both containing the parameters
            of the latent distributions as values.
        """
        # Reconstruction loss is pixel wise cross-entropy
        recon_loss = F.binary_cross_entropy(recon_data.view(-1, self.model.num_pixels),
                                            data.view(-1, self.model.num_pixels))
        # F.binary_cross_entropy takes mean over pixels, so unnormalise this
        recon_loss *= self.model.num_pixels

        # Calculate KL divergences
        kl_cont_loss = 0  # Used to compute capacity loss (but not a loss in itself)
        kl_disc_loss = 0  # Used to compute capacity loss (but not a loss in itself)
        cont_capacity_loss = 0
        disc_capacity_loss = 0

        if self.model.is_continuous:

            mu, logvar = latent_dist['cont']

            log_pz, log_qz, log_prod_qzi, log_q_zCx = self._get_log_pz_qz_prodzi_qzCx(latent_cont_sample, mu, logvar, self.full_data_size, self.is_mss)
            # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
            mi_loss = (log_q_zCx - log_qz).mean()
            # TC[z] = KL[q(z)||\prod_i z_i]
            tc_loss = (log_qz - log_prod_qzi).mean()
            # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
            dw_kl_loss = (log_prod_qzi - log_pz).mean()

            anneal_reg = (self._linear_annealing(0, 1, self.num_steps, self.steps_anneal)
                        if self.model.training else 1)

            # total loss
            kl_cont_loss =  self.alpha_loss * mi_loss + self.beta * tc_loss + anneal_reg * self.gamma * dw_kl_loss

        if self.model.is_discrete:
            # Calculate KL divergence
            kl_disc_loss = self._kl_multiple_discrete_loss(latent_dist['disc'])
            # Linearly increase capacity of discrete channels
            disc_min, disc_max, disc_num_iters, disc_gamma = \
                self.disc_capacity
            # Increase discrete capacity without exceeding disc_max or theoretical
            # maximum (i.e. sum of log of dimension of each discrete variable)
            disc_cap_current = (disc_max - disc_min) * self.num_steps / float(disc_num_iters) + disc_min
            disc_cap_current = min(disc_cap_current, disc_max)
            # Require float conversion here to not end up with numpy float
            disc_theoretical_max = sum([float(np.log(disc_dim)) for disc_dim in self.model.latent_spec['disc']])
            disc_cap_current = min(disc_cap_current, disc_theoretical_max)
            # Calculate discrete capacity loss
            disc_capacity_loss = disc_gamma * torch.abs(disc_cap_current - kl_disc_loss)

        # Calculate total kl value to record it
        kl_loss = kl_cont_loss + kl_disc_loss

        # Calculate total loss
        total_loss = recon_loss + kl_cont_loss + disc_capacity_loss

        # Record losses
        if self.model.training and self.num_steps % self.record_loss_every == 1:
            self.losses['recon_loss'].append(recon_loss.item())
            self.losses['kl_loss'].append(kl_loss.item())
            self.losses['loss'].append(total_loss.item())

        # To avoid large losses normalise by number of pixels
        return total_loss / self.model.num_pixels

    def _kl_normal_loss(self, mean, logvar):
        """
        Calculates the KL divergence between a normal distribution with
        diagonal covariance and a unit normal distribution.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (N, D) where D is dimension
            of distribution.

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (N, D)
        """
        # Calculate KL divergence
        kl_values = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        # Mean KL divergence across batch for each latent variable
        kl_means = torch.mean(kl_values, dim=0)
        # KL loss is sum of mean KL of each latent variable
        kl_loss = torch.sum(kl_means)

        # Record losses
        if self.model.training and self.num_steps % self.record_loss_every == 1:
            self.losses['kl_loss_cont'].append(kl_loss.item())
            for i in range(self.model.latent_spec['cont']):
                self.losses['kl_loss_cont_' + str(i)].append(kl_means[i].item())

        return kl_loss

    def _kl_multiple_discrete_loss(self, alphas):
        """
        Calculates the KL divergence between a set of categorical distributions
        and a set of uniform categorical distributions.

        Parameters
        ----------
        alphas : list
            List of the alpha parameters of a categorical (or gumbel-softmax)
            distribution. For example, if the categorical atent distribution of
            the model has dimensions [2, 5, 10] then alphas will contain 3
            torch.Tensor instances with the parameters for each of
            the distributions. Each of these will have shape (N, D).
        """
        # Calculate kl losses for each discrete latent
        kl_losses = [self._kl_discrete_loss(alpha) for alpha in alphas]

        # Total loss is sum of kl loss for each discrete latent
        kl_loss = torch.sum(torch.cat(kl_losses))

        # Record losses
        if self.model.training and self.num_steps % self.record_loss_every == 1:
            self.losses['kl_loss_disc'].append(kl_loss.item())
            for i in range(len(alphas)):
                self.losses['kl_loss_disc_' + str(i)].append(kl_losses[i].item())

        return kl_loss

    def _kl_discrete_loss(self, alpha):
        """
        Calculates the KL divergence between a categorical distribution and a
        uniform categorical distribution.

        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the categorical or gumbel-softmax distribution.
            Shape (N, D)
        """
        disc_dim = int(alpha.size()[-1])
        log_dim = torch.Tensor([np.log(disc_dim)])
        if self.use_cuda:
            log_dim = log_dim.cuda()
        # Calculate negative entropy of each row
        neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
        # Take mean of negative entropy across batch
        mean_neg_entropy = torch.mean(neg_entropy, dim=0)
        # KL loss of alpha with uniform categorical variable
        kl_loss = log_dim + mean_neg_entropy
        return kl_loss
    
    def _linear_annealing(self, init, fin, step, annealing_steps):
        """Linear annealing of a parameter."""
        if annealing_steps == 0:
            return fin
        assert fin > init
        delta = fin - init
        annealed = min(init + delta * step / annealing_steps, fin)
        return annealed
    
    def _get_log_pz_qz_prodzi_qzCx(self, latent_sample, mu, logvar, n_data, is_mss=False):
        batch_size, hidden_dim = latent_sample.shape

        # calculate log q(z|x)
        log_q_zCx = self._log_density_gaussian(latent_sample, mu, logvar).sum(dim=1)

        # calculate log p(z)
        # mean and log var is 0
        zeros = torch.zeros_like(latent_sample)
        log_pz = self._log_density_gaussian(latent_sample, zeros, zeros).sum(1)

        if not is_mss:
            log_qz, log_prod_qzi = self._minibatch_weighted_sampling(mu, logvar,
                                                                     latent_sample,
                                                                     n_data)

        else:
            log_qz, log_prod_qzi = self._minibatch_stratified_sampling(mu, logvar,
                                                                       latent_sample,
                                                                        n_data)

        return log_pz, log_qz, log_prod_qzi, log_q_zCx


    def _minibatch_weighted_sampling(self, mu, logvar, latent_sample, data_size):
        """
        Estimates log q(z) and the log (product of marginals of q(z_j)) with minibatch
        weighted sampling.

        Parameters
        ----------
        latent_dist : tuple of torch.tensor
            sufficient statistics of the latent dimension. E.g. for gaussian
            (mean, log_var) each of shape : (batch_size, latent_dim).

        latent_sample: torch.Tensor
            sample from the latent dimension using the reparameterisation trick
            shape : (batch_size, latent_dim).

        data_size : int
            Number of data in the training set

        References 
        -----------
        [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
        autoencoders." Advances in Neural Information Processing Systems. 2018.
        """
        batch_size = latent_sample.size(0)

        mat_log_qz = self._matrix_log_density_gaussian(latent_sample, mu, logvar)

        log_prod_qzi = (torch.logsumexp(mat_log_qz, dim=1, keepdim=False) -
                        math.log(batch_size * data_size)).sum(dim=1)
        log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False
                                ) - math.log(batch_size * data_size)

        return log_qz, log_prod_qzi


    def _minibatch_stratified_sampling(self, mu, logvar, latent_sample, data_size):
        """
        Estimates log q(z) and the log (product of marginals of q(z_j)) with minibatch
        stratified sampling.
        
        Parameters
        -----------
        latent_dist : tuple of torch.tensor
            sufficient statistics of the latent dimension. E.g. for gaussian
            (mean, log_var) each of shape : (batch_size, latent_dim).

        latent_sample: torch.Tensor
            sample from the latent dimension using the reparameterisation trick
            shape : (batch_size, latent_dim).

        data_size : int
            Number of data in the training set

        References 
        -----------
        [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
        autoencoders." Advances in Neural Information Processing Systems. 2018.
        """
        batch_size = latent_sample.size(0)

        mat_log_qz = self._matrix_log_density_gaussian(latent_sample, mu, logvar)

        log_iw_mat = self._log_importance_weight_matrix(batch_size, data_size).to(latent_sample.device)
        log_qz = torch.logsumexp(log_iw_mat + mat_log_qz.sum(2), dim=1, keepdim=False)
        log_prod_qzi = torch.logsumexp(log_iw_mat.view(batch_size, batch_size, 1) +
                                    mat_log_qz, dim=1, keepdim=False).sum(1)

        return log_qz, log_prod_qzi

    def _matrix_log_density_gaussian(self, x, mu, logvar):
        """Calculates log density of a Gaussian for all combination of bacth pairs of
        `x` and `mu`. I.e. return tensor of shape `(batch_size, batch_size, dim)`
        instead of (batch_size, dim) in the usual log density.

        Parameters
        ----------
        x: torch.Tensor
            Value at which to compute the density. Shape: (batch_size, dim).

        mu: torch.Tensor
            Mean. Shape: (batch_size, dim).

        logvar: torch.Tensor
            Log variance. Shape: (batch_size, dim).

        batch_size: int
            number of training images in the batch
        """
        batch_size, dim = x.shape
        x = x.view(batch_size, 1, dim)
        mu = mu.view(1, batch_size, dim)
        logvar = logvar.view(1, batch_size, dim)
        return self._log_density_gaussian(x, mu, logvar)

    def _log_density_gaussian(self, x, mu, logvar):
        """Calculates log density of a Gaussian.

        Parameters
        ----------
        x: torch.Tensor or np.ndarray or float
            Value at which to compute the density.

        mu: torch.Tensor or np.ndarray or float
            Mean.

        logvar: torch.Tensor or np.ndarray or float
            Log variance.
        """
        normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
        inv_var = torch.exp(-logvar)
        log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)
        return log_density

    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        """
        Calculates a log importance weight matrix

        Parameters
        ----------
        batch_size: int
            number of training images in the batch

        dataset_size: int
        number of training images in the dataset
        """
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M + 1] = 1 / N
        W.view(-1)[1::M + 1] = strat_weight
        W[M - 1, 0] = strat_weight
        return W.log()
