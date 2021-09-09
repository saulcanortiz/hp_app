import torch

class NetArchitecture(torch.nn.Module):
    def __init__(self):
        super(NetArchitecture, self).__init__()
        self.layer01 = torch.nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1)
        self.layer02 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.layer03 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.layer04 = torch.nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1)
        self.layer05 = torch.nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1)
        self.layer06 = torch.nn.Conv2d(160, 32, kernel_size=3, stride=1, padding=1)
        self.layer07 = torch.nn.Conv2d(192, 32, kernel_size=3, stride=1, padding=1)
        self.layer08 = torch.nn.Conv2d(224, 32, kernel_size=3, stride=1, padding=1)
        self.layerout = torch.nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1)
        


   
    def forward(self, x):
        x1 = torch.relu(self.layer01(x))
        x2 = torch.relu(self.layer02(x1))
        c2_dense = torch.relu(torch.cat([x1, x2], 1))
        
        x3 = torch.relu(self.layer03(c2_dense))
        c3_dense = torch.relu(torch.cat([x1, x2, x3], 1))
        
        x4 = torch.relu(self.layer04(c3_dense))
        c4_dense = torch.relu(torch.cat([x1, x2, x3, x4], 1))
        
        x5 = torch.relu(self.layer05(c4_dense))
        c5_dense = torch.relu(torch.cat([x1, x2, x3, x4, x5], 1))
        
        x6 = torch.relu(self.layer06(c5_dense))
        c6_dense = torch.relu(torch.cat([x1, x2, x3, x4, x5, x6], 1))
        
        x7 = torch.relu(self.layer07(c6_dense))
        c7_dense = torch.relu(torch.cat([x1, x2, x3, x4, x5, x6,x7], 1))
        
        x8 = torch.relu(self.layer08(c7_dense))
        c8_dense = torch.relu(torch.cat([x1, x2, x3, x4, x5, x6,x7,x8], 1))
        
        x9 = torch.relu(self.layerout(c8_dense))
        
        return x9