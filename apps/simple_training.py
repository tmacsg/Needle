import sys
sys.path.append('./python')
sys.path.append('./utils')
import needle as ndl
import needle.nn as nn
from needle import backend_ndarray as nd
from models import *
import time
import yaml
import os
from needle.data import *
from tqdm import tqdm

device = ndl.cpu()

### CIFAR-10 training ###

def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION



### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def train_fetal(model, optimizer, train_loader, test_loader, num_epochs, device):
    """
    
    """
    to_track = ["epoch", "train loss", "test loss"]

    results = {}
    for item in to_track:
        results[item] = []          

    for epoch in tqdm(range(num_epochs), desc="Epoch"):   
        results["epoch"].append(epoch)    
        train_loss, test_loss = run_fetal_epoch(model, optimizer, train_loader, test_loader, device)
        cur_train_loss, cur_test_loss = np.mean(train_loss), np.mean(test_loss)
        results['train loss'].append(cur_train_loss)
        results['test loss'].append(cur_test_loss)
        print(f'Epoch: {epoch}, train loss: {cur_train_loss}, test loss: {cur_test_loss}. ')

    return results

def run_fetal_epoch(model, optimizer, train_loader, test_loader, device):
    """ 
    """
    model.train()
    train_loss = []
    for images, masks in train_loader:
        optimizer.reset_grad()
        X, y = ndl.Tensor(images, device=device), ndl.Tensor(masks, device=device)        
        out = model(X)
        B,C,H,W = out.shape
        loss = nn.SoftmaxLoss()(out.transpose((1,2)).transpose((2,3)).reshape((B*H*W, C)), 
            y.reshape((B*H*W,)))
        loss.backward()
        optimizer.step()    
        train_loss.append(loss.detach().numpy()[0])
        # print('One batch done, loss: ', train_loss[-1])


    model.eval()
    test_loss = []
    for images, masks in test_loader:
        X, y = ndl.Tensor(images, device=device), ndl.Tensor(masks, device=device)        
        out = model(X)
        B,C,H,W = out.shape
        loss = nn.SoftmaxLoss()(out.transpose((1,2)).transpose((2,3)).reshape((B*H*W, C)), 
            y.reshape((B*H*W,)))        
        test_loss.append(loss.numpy()[0])

    return train_loss, test_loss


if __name__ == "__main__":
    ### For testing purposes
    #dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    #dataloader = ndl.data.DataLoader(\
    #         dataset=dataset,
    #         batch_size=128,
    #         shuffle=True
    #         )
    #
    #model = ResNet9(device=device, dtype="float32")
    #train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
    #      lr=0.001, weight_decay=0.001)

    # corpus = ndl.data.Corpus("./data/ptb")
    # seq_len = 40
    # batch_size = 16
    # hidden_size = 100
    # train_data = ndl.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
    # model = LanguageModel(1, len(corpus.dictionary), hidden_size, num_layers=2, device=device)
    # train_ptb(model, train_data, seq_len, n_epochs=10, device=device)


    configs = {}
    with open('./config/fetal.yaml') as f:
        configs = configs | yaml.safe_load(f)

    li = os.listdir(configs['data_path'] + '/all_images/')
    train_image_count = int(configs['data_split'][0] * len(li))
    train_list = li[0: train_image_count]
    test_list = li[train_image_count:]
    train_loader = DataLoader(FetalHeadDataset(configs['data_path'], train_list), batch_size=configs['batch_size'])
    test_loader = DataLoader(FetalHeadDataset(configs['data_path'], test_list))

    device = ndl.cuda() if configs['device'] == 'cuda' else ndl.cpu()
    model = unet(feature_scale=configs['feature_scale'], in_channels=1, n_classes=2, 
                device=device, dtype="float32", is_batchnorm=configs['is_batchnorm'])
    optimizer = ndl.optim.Adam(model.parameters(), lr=configs['lr'], weight_decay=configs['wt_dec'])

    train_fetal(model, optimizer, train_loader, test_loader, configs['num_epochs'], device)