import builtins

import nvsmi
import threading
# pynvml.nvmlInit()
from torch_geometric.datasets import Planetoid
import torch
import time
import numpy as np
import psutil
import os
from torch_geometric.datasets import Reddit
from tqdm import tqdm

dataset = Planetoid(root='./cora/', name='Cora')
# dataset = Planetoid(root='./citeseer',name='Citeseer')
# dataset = Planetoid(root='./pubmed/',name='Pubmed')
# dataset = Reddit(root='./reddit/')
print(dataset)

dataset = dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv

path = "/mnt/NVme/project_moka/pubmed/"
path = "/mnt/ramfs/project_moka/pubmed/"
# path = "/mnt/ext4ramdisk/project_moka/pubmed/"
# path = "./pubmed/"

path_Cora = "/mnt/NVme/project_moka/data/Cora/"
# path_Cora = "/mnt/ramfs/project_moka/data/Cora/"
# path_Cora = "/mnt/ext4ramdisk/project_moka/data/Cora/"
# path_Cora = "./data/Cora/"

path_pm = "/mnt/NVme/datasets/"
path_ram = "/mnt/ramfs/datasets/"

times = 15
total_time = 0
total_run_time = 0
batch_size = 128
epoch_num = 20

# pre-load Planetoid
# dataset_test = Planetoid(root="./cora/", name='Cora')
# dataset_test = Planetoid(root=path_Cora, name='Cora')
# the dataset for test is shown below (different from the above)
# dataset_test = Planetoid(root='./data/Cora/', name='Cora')
# dataset_test = Reddit(root='./reddit/')
#
# dataset = dataset_test

data = dataset[0]  # Get the first graph object.
from torch_geometric.data import ClusterData, ClusterLoader, DataLoader

torch.manual_seed(1234)
cluster_data = ClusterData(data, num_parts=128)  # 1. Create subgraphs.
train_loader = ClusterLoader(cluster_data, batch_size=batch_size,
                             shuffle=True)  # 2. Stochastic partitioning scheme.

print(train_loader)

class SAGENet(nn.Module):
    def __init__(self):
        super(SAGENet, self).__init__()
        self.conv1 = SAGEConv(dataset.num_node_features, 128)
        self.conv2 = SAGEConv(128, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.softmax(x, dim=1)

        return x


class GATNet(torch.nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()
        self.gat1 = GATConv(dataset.num_node_features, 8, 8, dropout=0.6)
        self.gat2 = GATConv(64, 7, 1, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = self.gat2(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


class GCNNet(torch.nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)

        return x


model = SAGENet()
print(model)
# model = GATNet()
# print(model)
# model = GCNNet()
# print(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
data = dataset[0].to(device)
print(data)
import torch.optim as optim

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
model.train()

gpu_data = list(nvsmi.get_gpus())[0]
start_gpu_util = gpu_data.gpu_util
start_gpu_mem_use = gpu_data.mem_used
total_gpu_mem = gpu_data.mem_total
disk_io_counter = psutil.disk_io_counters()
disk_total = disk_io_counter[2] + disk_io_counter[3]  # read_bytes + write_bytes
p = psutil.Process()
group_mem_rss = []
group_gpu_util = []
group_gpu_mem_use = []
group_disk_usage = []
group_iowait = []
t_status = True


def get_gpu_info():

    while True:
        if not t_status:
            break
        gpu_data = list(nvsmi.get_gpus())[0]
        group_gpu_util.append(gpu_data.gpu_util)
        group_gpu_mem_use.append(gpu_data.mem_used)
        group_mem_rss.append(psutil.Process(os.getpid()).memory_info().rss)
        io_counters = p.io_counters()
        disk_usage = io_counters[2] + io_counters[3]
        group_disk_usage.append(disk_usage)  # read_bytes + write_bytes
        try:
            a = psutil.cpu_times_percent().iowait
        except:
            a = 0
        group_iowait.append(a)
        # print(f"{disk_usage / 1024 / 1024} Mb/s")
        time.sleep(0.09)

start_timme = time.time()
for epoch in range(200):
    if epoch == 1:
        t = threading.Thread(target=get_gpu_info)
        t.start()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # pbar = tqdm(total=int(len(train_loader)))
    # pbar.set_description(f'Epoch {epoch:02d}')
    # for data, i in train_loader, len(train_loader):
    #     data = data.to(device)
    #     model.train()
    #     out = model(data)
    #     optimizer.zero_grad()
    #     loss = criterion(model(data.x, data.edge_index), data.y)
    #     # total_loss += float(loss.item())
    #     loss.backward()
    #     optimizer.step()
    #     pbar.update(i)
    #
    #     _, pred = torch.max(out[data.train_mask], dim=1)
    #     correct = (pred == data.y[data.train_mask]).sum().item()
    #     # print(correct)
    #     acc = correct / data.train_mask.sum().item()
    #     # print(data.train_mask.sum())
    #
    #     print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f}'.format(
    #         epoch, loss.item(), acc))
    # pbar.close()
    _, pred = torch.max(out[data.train_mask], dim=1)
    correct = (pred == data.y[data.train_mask]).sum().item()
    # print(correct)
    acc = correct / data.train_mask.sum().item()
    # print(data.train_mask.sum())

    print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f}'.format(
        epoch, loss.item(), acc))

t_status = False
t.join()
end_time = time.time()
print(f"GPU 显存占用: {np.mean(group_gpu_mem_use)}Mb")
print(f"GPU 显存占用率: {np.mean(group_gpu_mem_use) * 100 / total_gpu_mem}%")
print(f"GPU 平均使用率: {np.mean(group_gpu_util) - start_gpu_util}%")
tmp_ = builtins.sum(np.where(np.array(group_gpu_util) - start_gpu_util > 1, True, False))
print(f"GPU 空闲率: {(len(group_gpu_util) - tmp_) * 100 / len(group_gpu_util)}%")
print(f'内存使用：{np.mean(group_mem_rss) / 1024 / 1024 / 1024:.4f} GB')
print(f'磁盘IO使用：{np.mean(group_disk_usage) / 1024 / 1024 / 1024:.4f} GB/s')
print(f'磁盘IO使用率：{np.mean(group_disk_usage) * 100 / disk_total:.4f}%')
print(f'cpu iowait：{np.mean(group_iowait)}')

out = model(data)
loss = criterion(out[data.test_mask], data.y[data.test_mask])
_, pred = torch.max(out[data.test_mask], dim=1)
correct = (pred == data.y[data.test_mask]).sum().item()
acc = correct / data.test_mask.sum().item()
print("test_loss: {:.4f} test_acc: {:.4f}".format(loss.item(), acc))
print("Total training time:", end_time -start_timme)