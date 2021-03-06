import torch as torch
import torch.nn as nn
import torch.optim as optim
from model.train import Train


def run():
    network = Train(data_type=torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor,
                    model_filename="model1.pt",
                    create_new=False,
                    print_every=20,
                    loader_params={
                        'batch_size': 256,
                        'num_workers': 8 if torch.cuda.is_available() else 0
                    })
    loss_fn = nn.CrossEntropyLoss().type(network.data_type)
    network.train(loss_fn, optim.Adam(network.model.parameters(), lr=1e-5, weight_decay=1e-3), num_epochs=8)

run()
