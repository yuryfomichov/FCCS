import torch as torch
import torch.nn as nn
import torch.optim as optim
from model.train import Train


def run():
    network = Train(data_type=torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor,
                    model_filename="model.pt",
                    create_new=False,
                    print_every=20,
                    loader_params={
                         'batch_size': 256,
                         'num_workers': 32 if torch.cuda.is_available() else 0
                    })
    loss_fn = nn.CrossEntropyLoss().type(network.data_type)
    network.train(loss_fn, optim.Adam(network.model.classifier.parameters(), lr=1e-4), num_epochs=1)
    network.train(loss_fn, optim.Adam(network.model.classifier.parameters(), lr=1e-5), num_epochs=1)
    network.train(loss_fn, optim.Adam(network.model.classifier.parameters(), lr=1e-6), num_epochs=1)

run()
