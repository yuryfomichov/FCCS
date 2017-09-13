import torch as torch
import torchvision.models as models
from .datasetloader import DatasetLoader
from torch.autograd import Variable

class NetworkModel(object):
    def __init__(self, data_type=torch.cuda.FloatTensor, model_filename='model.pt', create_new=False, print_every=10, loader_params={}):
        self.data_type = data_type
        self.model_filename = model_filename
        self.create_new = create_new
        self.print_every = print_every
        self.init()
        self.loader = DatasetLoader(loader_params)

    def init(self):
        if self.create_new:
            self.model = self.init_model()
        else:
            self.model = self.load_model()

    def init_model(self):
        model = models.resnet18(num_classes=91)
        # simple_model = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     Flatten(),
        #     nn.Linear(256, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 91)
        # )

        model = model.type(self.data_type)
        return model

    def save_model(self):
        torch.save(self.model_filename)

    def load_model(self):
        return torch.load(self.model_filename)

    def train(self, loss_fn, optimizer, num_epochs=1):
        for epoch in range(num_epochs):
            print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
            self.model.train()
            for t, (x, y) in enumerate(self.loader.get_train_loader()):
                x_var = Variable(x.type(self.data_type))
                y_var = Variable(y.type(self.data_type).long())

                scores = self.model(x_var)

                loss = loss_fn(scores, y_var)
                if (t + 1) % self.print_every == 0:
                    print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.save_model()
            self.check_val_accuracy()

    def check_val_accuracy(self):
        print('Checking accuracy on Validation set')
        self.check_accuracy(self.loader.get_val_loader())

    def check_test_accuracy(self):
        print('Checking accuracy on Test set')
        self.check_accuracy(self.loader.get_test_loader())

    def check_accuracy(self, loader):
        num_correct = 0
        num_samples = 0
        self.model.eval()
        for x, y in loader:
            x_var = Variable(x.type(self.data_type), volatile=True)

            scores = self.model(x_var)
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))