import torch
import torchvision
import torch.nn as nn
from LeNet_model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms

def main():
    transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10( root = './data', train = True, download = True, transform = transform )
    trainloader = torch.utils.data.DataLoader( trainset, batch_size = 36, shuffle = True, num_workers =0 )

    testset = torchvision.datasets.CIFAR10( root = './data', train = False, download = False, transform = transform )
    testloader = torch.utils.data.DataLoader( testset, batch_size = 5000, shuffle = False, num_workers =0 )

    test_data_iter = iter( testloader )
    test_images, test_labels = next( test_data_iter )

    net = LeNet()
    loss_f = nn.CrossEntropyLoss()  # 损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # lr -> learning rate

    # 迭代训练集
    for ep in range(5):
        running_loss = 0.0
        for step, data in enumerate(trainloader, start=0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_f(outputs, labels)
            loss.backward()  # 反向传播
            optimizer.step()

            running_loss += loss.item()
            if step % 500 == 499:  # 每500步打印
                with torch.no_grad():  # 测试过程中不计算每个节点的误差损失梯度
                    outputs = net(test_images)
                    predict_y = torch.max(outputs.data, dim=1)[1]
                    accuracy = torch.eq(predict_y, test_labels).sum().item() / test_labels.size(0)
                    print('[%d, %5d] loss:%.3f accuracy:%.3f' % (ep + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    save_path = './path/LeNet.pth'
    torch.save(net.state_dict(), save_path)
    print('finished')

if __name__ == '__main__':
    main()