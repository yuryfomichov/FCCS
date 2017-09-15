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
            transforms.Scale(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            ToCudaTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        return transform


class ToCudaTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        img = torch.cuda.ByteTensor(torch.cuda.ByteStorage.from_buffer(pic.tobytes()))
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.cuda.ByteTensor):
            return img.float().div(255)
        else:
            return img