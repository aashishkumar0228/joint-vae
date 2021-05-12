import torch
from jointvae.models import VAE
from jointvae.models_custom import BTCVAE_EMNIST
from jointvae.training import Trainer
from jointvae.training_btcvae import Trainer_BTCVAE
from utils.dataloaders import get_mnist_dataloaders, get_emnist_uppercase_dataloaders, get_emnist_uppercase_reduced_dataloaders
from torch import optim


batch_size = 64
lr = 5e-4
epochs = 100

# Check for cuda
use_cuda = torch.cuda.is_available()
# use_cuda = False

# Load data
# data_loader, _ = get_mnist_dataloaders(batch_size=batch_size, path_to_data='/home/kaushikdas/aashish/pytorch_datasets')
data_loader, _ = get_emnist_uppercase_dataloaders(batch_size=batch_size,
                                                  path_to_train_csv='/home/kaushikdas/aashish/pytorch_datasets/EMNIST_UPPERCASE_LETTER/emnist_uppercase_train_4th_May_2021.csv',
                                                  path_to_test_csv='/home/kaushikdas/aashish/pytorch_datasets/EMNIST_UPPERCASE_LETTER/emnist_uppercase_test_3rd_May_2021.csv')

# data_loader, _ = get_emnist_uppercase_reduced_dataloaders(batch_size=batch_size,
#                                                   path_to_train_csv='/home/kaushikdas/aashish/pytorch_datasets/EMNIST_UPPERCASE_LETTER_REDUCED/emnist_uppercase_train_11th_May_2021_reduced.csv',
#                                                   path_to_test_csv='/home/kaushikdas/aashish/pytorch_datasets/EMNIST_UPPERCASE_LETTER_REDUCED/emnist_uppercase_test_11th_May_2021_reduced.csv')


img_size = (1, 32, 32)

# Define latent spec and model
# latent_spec = {'cont': 10, 'disc': [10]}
latent_spec = {'cont': 10}
hidden_dim = 256
beta = 50
is_mss = True
model = BTCVAE_EMNIST(img_size=img_size, latent_spec=latent_spec,
            use_cuda=use_cuda, hidden_dim=hidden_dim)
if use_cuda:
    model.cuda()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Define trainer
trainer = Trainer_BTCVAE(model, optimizer,
                  cont_capacity=[0.0, 5.0, 25000, 30.0],
                  disc_capacity=[0.0, 5.0, 25000, 30.0],
                  use_cuda=use_cuda,
                  beta=beta
                  is_mss=is_mss)

# Train model for 100 epochs
trainer.train(data_loader, epochs)

# Save trained model
torch.save(trainer.model.state_dict(), 'example-model.pt')
