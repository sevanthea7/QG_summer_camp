import torch
import torchvision.transforms as transforms
from PIL import Image
from LeNet_model import LeNet


def main():
    transform = transforms.Compose( [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    net = LeNet()
    net.load_state_dict(torch.load('./path/Lenet.pth'))

    im = Image.open('./data/test3.jpg')
    im = transform(im)  # 高度，宽度，深度
    im = torch.unsqueeze(im, dim=0)  # 增加新的维度：batch，高度，宽度，深度

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].numpy()
    print( classes[int(predict)] )


if __name__ == '__main__':
    main()