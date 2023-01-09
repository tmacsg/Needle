import sys
sys.path.append('./python')
import numpy as np
import pytest
from needle import backend_ndarray as nd
import needle as ndl
from needle.autograd import TENSOR_COUNTER
from needle.data import *
from needle import nn
import torch

TEST_SIZE = 128

class toy_dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.X = np.random.rand(32,1,TEST_SIZE,TEST_SIZE).astype(np.float32)
        self.y = (np.random.rand(32,1,TEST_SIZE,TEST_SIZE) > 0.5).astype(np.int32)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.X.shape[0]

class toy_model(nn.Module):
    def __init__(self, device):
        self.device = device
        self.model = nn.Sequential(
            nn.Conv(2,2,1,padding=0,device=device),  #(128,2,8,8)       
            nn.Maxpool(2), #(128,2,4,4)   
            nn.ReLU(),
            nn.Conv(2,2,1,padding=0,device=device),  #(128,2,4,4)          
            nn.Maxpool(2), #(128,2,2,2) 
            nn.ReLU(),
            nn.Conv_transposed(2,2,2,2,padding=0, device=device), #(128,2,4,4) 
            nn.ReLU(),
            nn.Conv(2,1,1,padding=0,device=device),  #(128,1,4,4)
        )

    def forward(self, x):
        x1 = ndl.ops.unpad(x, ((0,0),(0,0),(32,32),(32,32))) #(128,2,4,4)
        x2 = nn.Conv(2,1,1,padding=0,device=self.device)(x1) #(128,1,4,4)
        x3 = self.model(x) #(128,1,4,4)
        x4 = ndl.ops.concat([x2,x3],1) #(128,2,4,4)
        return x4

class toy_model_v2(nn.Module):
    def __init__(self, device):
        self.device = device
        self.model = nn.Sequential(
            nn.Conv(2,4,1,padding=0,device=device, bias=True),  #(128,2,8,8)   
            nn.BatchNorm2d(dim=4, device=device, dtype="float32"),
            nn.ReLU(),
            nn.Conv(4,2,1,padding=0,device=device, bias=True),
        )

    def forward(self, x):
        x = self.model(x) #(128,1,4,4)
        return x

DEVICE = ndl.cuda()
def test_nn_modules():
    data_loader = DataLoader(toy_dataset(), batch_size=16)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(TEST_SIZE*TEST_SIZE, TEST_SIZE*TEST_SIZE, device=DEVICE, bias=False),
        # nn.Identity(),
        nn.ReLU(),
        # nn.Linear(128*128,128*128, device=ndl.cuda())
    )
    opt = ndl.optim.SGD(params=model.parameters())
    model.train()

    # m = ndl.Tensor(np.random.rand(10000,10000), 
    #                 device=DEVICE)
    # n = ndl.Tensor(np.random.rand(10000,10000), 
    #                 device=DEVICE)
    # test_opt = ndl.optim.SGD(params=[m])  

    for _ in range(10):
        cur_loss = 0 
        for data, label in data_loader:  
            opt.reset_grad()          
            X, y = ndl.Tensor(data, device=DEVICE), ndl.Tensor(label, device=DEVICE)
            y_hat = model(X)       
            y = y.reshape(y_hat.shape)
            loss = nn.MSELoss()(y_hat, y)  
            loss.backward()        
            cur_loss += loss.data.numpy()[0]
            opt.step()
            # m = np.random.rand(10000,10000)
            # n = m @ m
            # print(n.shape)
            
        print('Loss: ', cur_loss)

def test_nn_modules_torch():
    data_loader = DataLoader(toy_dataset(), batch_size=32)
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(TEST_SIZE*TEST_SIZE, TEST_SIZE*TEST_SIZE, bias=False),
        torch.nn.ReLU(),
        # nn.Linear(128*128,128*128, device=ndl.cuda())
    )
    
    model.cuda()
    opt = torch.optim.SGD(params=model.parameters(), 
        lr=0.01, weight_decay=0.009)
    model.train()
    for _ in range(50):
        cur_loss = 0 
        for data, label in data_loader:
            opt.zero_grad()
            X, y = torch.Tensor(data).cuda(), \
                    torch.Tensor(label).cuda()
            y_hat = model(X)
        
            y = y.reshape(y_hat.shape)

            loss = torch.nn.MSELoss()(y_hat, y)  
            loss.backward()        
            cur_loss += loss.cpu().item()
            opt.step()
        print('Loss: ', cur_loss)
    


def test_toy_dataset():
    data_set = toy_dataset()
    batch_size = 128
    n_batch = len(data_set) // batch_size
    epochs = 50

    data_loader = DataLoader(data_set, batch_size=batch_size)
    model = toy_model_v2(device=ndl.cuda())
    opt = ndl.optim.Adam(params=model.parameters())

    model.train()
    losses = []
    for epoch in range(epochs):        
        cur_loss = 0 
        for data, label in data_loader:
            opt.reset_grad()
            X, y = ndl.Tensor(data, device=ndl.cuda()), ndl.Tensor(label, device=ndl.cuda())
            y_hat = model(X)
            B,C,H,W = y_hat.shape
            loss = nn.SoftmaxLoss()(y_hat.transpose((1,2)).transpose((2,3)).reshape((B*H*W, C)),
                                y.transpose((1,2)).transpose((2,3)).reshape((B*H*W,)))   
            loss.backward()        
            cur_loss += loss.data.numpy()[0]
            opt.step()
        print(f'Epoch {epoch}: {cur_loss / n_batch}')
        losses.append(cur_loss)

        
