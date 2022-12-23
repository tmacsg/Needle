import sys
sys.path.append('./python')
sys.path.append('./apps')
from needle import ops
import needle as ndl
import numpy as np
from needle import backend_ndarray as array_api

# from tests.test_sequence_models import *
# from tests.test_conv import *

# _DEVICES = [ndl.cpu(), pytest.param(ndl.cuda(),
#     marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]


def one_iter_of_cifar10_training(dataloader, model, niter=1, loss_fn=ndl.nn.SoftmaxLoss(), opt=None, device=None):
    np.random.seed(4)
    model.train()
    params_start = model.parameters()
    correct, total_loss = 0, 0
    i = 1
    for batch in dataloader:
        opt.reset_grad()
        X, y = batch
        X, y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
        out = model(X)
        correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
        loss = loss_fn(out, y)
        total_loss += loss.data.numpy() * y.shape[0]
        loss.backward()
        opt.step()
        if i >= niter:
            break
        i += 1
    # print('loss: ', correct/(y.shape[0]*niter), total_loss/(y.shape[0]*niter))
    params_end = model.parameters()
    print(params_end == params_start)
    return correct/(y.shape[0]*niter), total_loss/(y.shape[0]*niter)

if __name__ == '__main__':
    # Z_shape = ((2,3,4,5),(2,3,4,5))
    # axis = 0
    # test_op_concat(Z_shape, axis, False, _DEVICES[0])
    import sys
    sys.path.append('.')
    from apps.models import ResNet9
    np.random.seed(1)
    device = ndl.cuda()
    model = ResNet9(device=device)
    
    # for param in model.parameters():
    #     print(type(param))
    
    ndl.nn.save(model.parameters(), 'test.pkl')

    np.random.seed(1)
    dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    dataloader = ndl.data.DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=True
    )
    # np.random.seed(1)
    # model = ResNet9(device=device, dtype="float32")
    # out = one_iter_of_cifar10_training(dataloader, model, niter=1, opt=ndl.optim.Adam(
    #     model.parameters(), lr=0.01, weight_decay=0.0001), device=device)
    
    # print(out)
    