import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()      #解决继承过程中的问题
        self.conv1 = nn.Conv2d( 3, 16, 5 )
        self.pool1 = nn.MaxPool2d( 2, 2 )   # 改变高和宽，不影响深度 eg. 16, 28, 28 -> 16, 14, 14
        self.conv2 = nn.Conv2d( 16, 32, 5 ) # 第2个conv输出的卷积核深度为16，到第2个卷积核深度
        self.pool2 = nn.MaxPool2d( 2, 2 )
        self.fc1 = nn.Linear( 32*5*5, 120 )
        self.fc2 = nn.Linear( 120, 84 )
        self.fc3 = nn.Linear( 84, 10 )
    def forward(self, x):
        x = F.relu( self.conv1( x ) )       # in: 3, 32, 32  out: 16, 28, 28
        x = self.pool1( x )                 # out: 16, 14, 14
        x = F.relu( self.conv2( x ) )       # out: 32, 10, 10
        x = self.pool2( x )                 # out: 32, 5, 5
        x = x.view( -1, 32*5*5 )            # out: 32*5*5
        x = F.relu( self.fc1( x ) )         # out: 120
        x = F.relu( self.fc2( x ) )         # out: 84
        x = self.fc3( x )                   # out: 10
        return x