import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.dataloader as dataloader
import torch as torch


class DatasetLoader(object):
    def __init__(self, params):
        self.data_dir = 'datasets'
        self.train_dir = 'train2017'
        self.val_dir = 'val2017'
        self.test_dir = 'test2017'
        self.batch_size = params.get("batch_size", 200)
        self.num_workers = params.get("num_workers", 32)
        self.cache = {}

    def get_loader(self, dataType, drop_last = False):
        annFile = '%s/annotations/instances_%s.json' % (self.data_dir, dataType)
        dataFolder = '%s/%s/' % (self.data_dir, dataType)

        data = self.cache.get(annFile)
        if data is None:
            data = datasets.CocoDetection(dataFolder, annFile, self.input_transform(), self.target_transform)
            self.cache[annFile] = data

        loader = dataloader.DataLoader(data,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       num_workers=self.num_workers,
                                       drop_last=drop_last,
                                       pin_memory=torch.cuda.is_available())
        return loader

    def get_train_loader(self, drop_last = True):
        return self.get_loader(self.train_dir, drop_last)

    def get_val_loader(self):
        return self.get_loader(self.val_dir)

    def get_test_loader(self):
        return self.get_loader(self.test_dir)

    def target_transform(self, annotation):
        result = 0
        if (len(annotation) > 0):
            result = max(list(map(lambda x: (x['area'], x['category_id']), annotation)), key=lambda item: item[0])[1]
        return result

    def input_transform(self):
        transform = transforms.Compose([
            transforms.Scale(152),
            transforms.RandomCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        return transform
