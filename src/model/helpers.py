import torchvision.transforms as transforms


def convert_tensor_to_image(tensor):
    preprocess2 = transforms.Compose([transforms.ToPILImage()])
    result_image = preprocess2(tensor)
    result_image.save('test.jpg', "JPEG")
    return result_image
