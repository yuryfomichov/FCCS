from pycocotools.coco import COCO
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.dataloader as dataloader
import torch as torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from PIL import Image

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)

data_type = torch.cuda.FloatTensor

def test_transform(dataDir, dataType):
    annFile = '%s/annotations/instances_%s.json' % (dataDir, dataType)
    dataFolder = '%s/%s/' %  (dataDir, dataType)
    coco = COCO(annFile)
    imgIds = coco.getImgIds()
    img = coco.loadImgs(imgIds[0])

    input_image = Image.open(dataFolder + img[0]['file_name'])
    annIds = coco.getAnnIds(img[0]['id'])
    anns = coco.loadAnns(annIds)

    preprocess = transforms.Compose([
    transforms.Scale(80),
    transforms.RandomCrop(64),
    transforms.ToTensor()])

    image_tensor = preprocess(input_image)
    return image_tensor

def load_anatation(dataDir, dataType):
    annFile = '%s/annotations/instances_%s.json' % (dataDir, dataType)
    coco = COCO(annFile)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    img = coco.loadImgs(imgIds)
    annIds = coco.getAnnIds(imgIds, catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    return coco

class Annotation_transform(object):
    def __init__(self):
        pass

    def __call__(self, annotation):
        result = 0
        if (len(annotation) > 0):
            result = max(list(map(lambda x: (x['area'], x['category_id']) , annotation)), key = lambda item: item[0])[1]
        return result

def get_loader(dataDir, dataType):
    annFile = '%s/annotations/instances_%s.json' % (dataDir, dataType)
    dataFolder = '%s/%s/' %  (dataDir, dataType)

    input_transform = transforms.Compose([
    transforms.Scale(128),
    transforms.CenterCrop(128),
    transforms.ToTensor()])

    target_transform = Annotation_transform()

    data = datasets.CocoDetection(dataFolder, annFile, input_transform, target_transform)
    loader = dataloader.DataLoader(data, batch_size=200, shuffle=True, num_workers=32)
    loader.dataset.train = True
    return loader

def save_model(model):
    torch.save(model, 'model.pt')

def load_model():
    return torch.load('model.pt')


def get_model():
    simple_model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        Flatten(),
        nn.Linear(4096, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 91)
    )

    simple_model = simple_model.type(data_type)
    return simple_model

def convert_tensor_to_image(tensor):
    preprocess2 = transforms.Compose([transforms.ToPILImage()])
    result_image = preprocess2(tensor)
    # result_image.save('test.jpg', "JPEG")
    return result_image


def train(model, loss_fn, optimizer, num_epochs=1):
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        for t, (x, y) in enumerate(get_loader('datasets/', 'train2017')):
            x_var = Variable(x.type(data_type))
            y_var = Variable(y.type(data_type).long())

            scores = model(x_var)

            loss = loss_fn(scores, y_var)
            if (t + 1) % 10 == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        save_model(model);
        check_accuracy(model, get_loader('datasets/', 'val2017'))

def check_accuracy(model, loader):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        x_var = Variable(x.type(data_type), volatile=True)

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)

    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


def main():
    model = load_model()
    loss_fn = nn.CrossEntropyLoss().type(data_type)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train(model, loss_fn, optimizer, num_epochs=3)

main()
