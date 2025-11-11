import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3, dropout_prob=0.3):
        super(cls_model, self).__init__()
        
        self.layers = []

        # Shared MLP: nx3 -> nx64
        self.first_block = [
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        ]

        self.second_block = [
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        ]

        self.final_block = [
            #TODO add dropout
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(256, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.ReLU(inplace=True),
        ]

        self.first_net = nn.Sequential(*self.first_block)
        self.second_net = nn.Sequential(*self.second_block)
        self.final_net = nn.Sequential(*self.final_block)


    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        B, N, _ = points.size()
        # x = points

        # x = points.view(B * N, 3)
        x = points.reshape((B*N, 3))
        x = self.first_net(x)
        x = self.second_net(x)
        x, _ = torch.max(x.view(B, N, -1), dim=1) #maxpool
        # x, _ = torch.max(x.reshape(B, N, -1), dim=1)
        x = self.final_net(x)
        return x.view(B, -1)


# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        
        self.layers = []
        self.m = num_seg_classes

        # Shared MLP: nx3 -> nx64
        self.first_block = [
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        ]

        self.second_block = [
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        ]

        self.seg_block = [
            nn.Linear(1088, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.m),
            nn.BatchNorm1d(self.m),
            nn.ReLU(inplace=True),
        ]


        self.first_net = nn.Sequential(*self.first_block)
        self.second_net = nn.Sequential(*self.second_block)
        self.seg_net = nn.Sequential(*self.seg_block)

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        B, N, _ = points.size()
        # x = points
        x = points.view(B * N, 3)
        local_feats = self.first_net(x)
        x = self.second_net(local_feats)
        global_feats, _ = torch.max(x.view(B, N, -1), dim=1) #maxpool
        global_feats = global_feats.repeat(N, 1)
        total_feats = torch.cat([local_feats, global_feats], dim=1)
        assert(total_feats.shape == (B*N, 1088))
        scores = self.seg_net(total_feats)
        scores = scores.view(B, N, self.m)
        assert(scores.shape == (B, N, self.m))

        return scores



