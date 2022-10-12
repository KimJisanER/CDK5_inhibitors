dir_data = '/home/kimjisan/PBI/outputs/rgcn'   # tgm.nn.data.Data들을 저장했던 폴더를 저장해주세요.
seed = 1                # 랜덤 시드 넘버

# Base model
# model architecture
num_node_features = 13  # 원자개수 (RGCN의 입력 노드 특징 벡터의 차원)
node_embedding_dim = 256     # 각 원자들을 나타내는 임베딩 벡터의 차원 수
hidden_channels = (256, 256, 256, 256, 256, 256, 256, 256)  # 각 RGCN 레이어가 만드는 hidden state 벡터의 차원 수
hidden_dims = (1024, 512)  # 각 MLP 레이어가 만드는 hidden state 벡터의 차원 수
dropout = 0.3

# training
lr = 0.0001
n_epochs = 1000
batch_size = 128
from glob import glob  # 폴더 안에 있는 파일들의 이름을 리스트 형태로 만들어주는 모듈

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import RGCNConv, global_add_pool
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader


class GNNDataset(InMemoryDataset):
    """
    폴더 안에 있는 tgm.nn.data.Data를 불러오는 함수
    """

    def __init__(self, root):
        super(GNNDataset, self).__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = glob(f'{self.root}/*.pt')
        data_list = list(map(torch.load, data_list))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

torch.manual_seed(1)

dataset = GNNDataset(f'{dir_data}/CHEMBL')

train_dataset = dataset[:1100]
valid_dataset = dataset[1100:]
test_dataset = GNNDataset(f'{dir_data}/ZINC')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


class NodeEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(NodeEncoder, self).__init__()
        self.encoder = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        return self.encoder(x)


class RGCNSkipConnection(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_dims, num_node_features=13, node_embedding_dim=256, dropout=0.3):
        super(RGCNSkipConnection, self).__init__()
        self.node_encoder = NodeEncoder(num_node_features, node_embedding_dim)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

        # GCN
        self.conv_layers = nn.ModuleList()
        for i in range(len(hidden_channels)):
            if i == 0:
                conv = RGCNConv(node_embedding_dim, hidden_channels[i], num_relations=6, aggr='add')
            else:
                conv = RGCNConv(hidden_channels[i - 1], hidden_channels[i], num_relations=6, aggr='add')
            self.conv_layers.append(conv)

        self.graph_pooling = nn.Sequential(
            nn.Linear(hidden_channels[-1], hidden_channels[-1]),
            nn.ReLU(),
            nn.Linear(hidden_channels[-1], hidden_channels[-1]),
        )

        # MLP
        self.fc_layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:
                fc = nn.Linear(hidden_channels[-1], hidden_dims[i])
            else:
                fc = nn.Linear(hidden_dims[i - 1], hidden_dims[i])
            self.fc_layers.append(fc)
        self.out = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x, edge_index, edge_type, batch):
        # 1. Encode each atom to a vector representation
        x = self.node_encoder(x)

        # 2. Obtain node embeddings through gnn
        for i, conv in enumerate(self.conv_layers):
            skip = x
            x = conv(x, edge_index, edge_type)
            x = self.prelu(x + skip)
            x = F.normalize(x, 2, 1)

        # 3. graph pooling
        x = self.graph_pooling(x)
        x = global_add_pool(x, batch)  # [batch_size, hidden_channels]

        # 4. Readout phase
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            x = self.dropout(x)
            x = F.relu(x)

        # 5. Final prediction
        x = F.relu(self.out(x))

        return x

model = RGCNSkipConnection(hidden_channels, hidden_dims, num_node_features, node_embedding_dim, dropout=dropout)

class AverageMeter:
    '''
    Compute and store the average and current value
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RGCNSolver:
    def __init__(self, model, lr, n_epochs, device=None):
        self.model = model
        self.device = device
        self.n_epochs = n_epochs

        self.criterion = torch.nn.L1Loss()
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.params, lr=lr)
        self.model.to(self.device)
        self.history = {'train_loss': [], 'valid_loss': []}

    def fit(self, train_loader, valid_loader):
        for epoch in range(self.n_epochs):
            # training phase
            t = time.time()
            train_loss = self.train_one_epoch(train_loader)

            # validation phase
            valid_loss = self.evaluate(valid_loader)


            message = f"[Epoch {epoch}] "
            message += f"Elapsed time: {time.time() - t:.3f} | "
            message += f"Train loss: {train_loss.avg:.5f} | "
            message += f"Validation loss: {valid_loss.avg:.5f} | "

            print(message)
            self.history['train_loss'].append(train_loss.avg)
            self.history['valid_loss'].append(valid_loss.avg)


    def train_one_epoch(self, loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()

        for step, data in enumerate(loader):
            print(
                f'Train step {(step + 1)} / {len(loader)} | ' +
                f'Summary loss: {summary_loss.avg:.5f} |' +
                f'Time: {time.time() - t:.3f} |', end='\r'
            )
            data.to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(data.x, data.edge_index, data.edge_type, data.batch)
            loss = self.criterion(y_pred, data.y.unsqueeze(1))
            loss.backward()
            self.optimizer.step()

            summary_loss.update(loss.detach().item(), data.num_graphs)

        return summary_loss

    def evaluate(self, loader):
        self.model.eval()
        summary_loss = AverageMeter()
        t = time.time()

        with torch.no_grad():
            for step, data in enumerate(loader):
                data.to(self.device)

                y_pred = self.model(data.x, data.edge_index, data.edge_type, data.batch)
                loss = self.criterion(y_pred, data.y.unsqueeze(1))
                summary_loss.update(loss.detach().item(), data.num_graphs)

        return summary_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
solver = RGCNSolver(model, lr, n_epochs, device)
solver.fit(train_loader, valid_loader)

import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, window_size=10):
    return [np.mean(x[i:i + window_size]) for i in range(len(x) - window_size + 1)]


def plotting_learning_curve(history, window_size=10):
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    epochs = np.arange(len(history['train_loss'])) + 1

    ax.plot(epochs[:-window_size + 1], moving_average(history['train_loss'], window_size), label='train_loss')
    ax.plot(epochs[:-window_size + 1], moving_average(history['valid_loss'], window_size), label='valid_loss')
    ax.grid()
    ax.legend()
    ax
    plt.show()


plotting_learning_curve(solver.history,window_size=10)

import pandas as pd

candidate = pd.read_csv('/home/kimjisan/PBI/data/test.csv')
candidate = candidate['uid']
candidate=pd.DataFrame(candidate)
candidate['pChEMBL Value']=''
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
model.eval()

preds = []
with torch.no_grad():
    for data in test_loader:
        data.to('cuda')
        pred = model(data.x, data.edge_index, data.edge_type, data.batch).detach().cpu().item()
        candidate.loc[candidate['uid'] == data.uid[0], 'pChEMBL Value'] = pred


candidate.to_csv('sub.csv', index=False)
