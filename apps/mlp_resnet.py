import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    fn = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim)
    )

    return nn.Sequential(
        nn.Residual(fn),
        nn.ReLU()
    )
    ### END YOUR SOLUTION

def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim // 2, norm=norm, drop_prob=drop_prob) 
            for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes)
    )
    ### END YOUR SOLUTION

def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
        model.eval()
    else:
        model.train()
    losses = 0
    err = 0
    count = 0
    count_2 = 0

    for x, y in dataloader:
        logits = model(x)
        loss = nn.SoftmaxLoss()(logits, y)      
        losses += loss.numpy()       
        y_predicted = np.argmax(logits.numpy(), 1)    
        y = y.numpy()
        err += np.sum(y_predicted != y)
        
        count += x.shape[0]
        count_2 += 1

        if model.training:  
            opt.reset_grad         
            loss.backward()
            opt.step()
    return err / count, losses / count_2
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(\
            data_dir + "/train-images-idx3-ubyte.gz",
            data_dir + "/train-labels-idx1-ubyte.gz")
    train_dataloader = ndl.data.DataLoader(\
             dataset=train_dataset,
             batch_size=batch_size,
             shuffle=True)

    test_dataset = ndl.data.MNISTDataset(\
            data_dir + "/t10k-images-idx3-ubyte.gz",
            data_dir + "/t10k-labels-idx1-ubyte.gz")
    test_dataloader = ndl.data.DataLoader(\
             dataset=test_dataset,
             batch_size=batch_size,
             shuffle=False)

    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_acc = 0
    train_loss = 0
    test_acc = 0
    test_loss = 0
    for _ in range(epochs):
        train_err, train_loss = epoch(train_dataloader, model, opt=opt)
        test_err, test_loss = epoch(test_dataloader, model, opt=None)
        
    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="./data")
