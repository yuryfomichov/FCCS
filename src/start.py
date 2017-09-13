import torch as torch
import torch.nn as nn
import torch.optim as optim
from model.model import NetworkModel

def run():
    network = NetworkModel(data_type=torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor,
                           model_filename="model.pt",
                           create_new=False,
                           print_every=20,
                           loader_params={
                               'batch_size': 300,
                               'num_workers': 32 if torch.cuda.is_available() else 0
                           })
    loss_fn = nn.CrossEntropyLoss().type(network.data_type)
    network.train(loss_fn, optim.Adam(network.model.parameters(), lr=1e-4), num_epochs=1)
    network.train(loss_fn, optim.Adam(network.model.parameters(), lr=5e-5), num_epochs=1)
    network.train(loss_fn, optim.Adam(network.model.parameters(), lr=1e-5), num_epochs=2)

run()