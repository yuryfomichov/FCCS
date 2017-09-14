import torch as torch
import torch.nn as nn
import time
from .model import Model
from .datasetloader import DatasetLoader
from torch.autograd import Variable

class Train(object):
    def __init__(self, data_type=torch.cuda.FloatTensor, model_filename='model.pt', create_new=False, print_every=10,
                 loader_params={}):
        self.data_type = data_type
        self.model_filename = model_filename
        self.create_new = create_new
        self.print_every = print_every
        self.init()
        self.loader = DatasetLoader(loader_params)

    def init(self):
        if self.create_new:
            print('New model was created')
            self.model = self.init_model()
        else:
            try:
                print('Model was loaded from file')
                self.model = self.load_model();
            except:
                print('No model had been found. New model was created')
                self.model = self.init_model()

    def init_model(self):
        model = Model()
        model = model.type(self.data_type)
        return model

    def save_model(self):
        torch.save(self.model, self.model_filename)

    def load_model(self):
        return torch.load(self.model_filename)

    def train(self, loss_fn, optimizer, num_epochs=1):
        for epoch in range(num_epochs):
            print('')
            print('--------------------------------------------------------------------------------------------------')
            print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
            tic = time.time()
            self.model.train()
            for t, (x, y) in enumerate(self.loader.get_train_loader()):
                x_var = Variable(x.type(self.data_type), requires_grad=False)
                y_var = Variable(y.type(self.data_type).long())
                scores = self.model(x_var)
                loss = loss_fn(scores, y_var)

                if (t + 1) % self.print_every == 0:
                    print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('Epoch done in t={:0.1f}s'.format(time.time() - tic))
            self.save_model()
            self.check_val_accuracy()
            self.check_train_accuracy()

    def check_train_accuracy(self):
        print('Checking accuracy on TRAIN set')
        return self.check_accuracy(self.loader.get_train_loader(), 5000)

    def check_val_accuracy(self):
        print('Checking accuracy on VALIDATION set')
        return self.check_accuracy(self.loader.get_val_loader())

    def check_test_accuracy(self):
        print('Checking accuracy on TEST set')
        return self.check_accuracy(self.loader.get_test_loader())

    def check_accuracy(self, loader, stop_on=-1):
        num_correct = 0
        num_samples = 0
        self.model.eval()
        for x, y in loader:
            x_var = Variable(x.type(self.data_type), volatile=True)
            scores = self.model(x_var)
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

            if (stop_on >= 0 and num_samples > stop_on):
                break;

        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return acc;
