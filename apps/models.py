import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.device=device

        def ConvBN(a,b,k,s, device=self.device):
            return nn.Sequential(
                nn.Conv(a, b, k, s, device=device, dtype="float32"),
                nn.BatchNorm2d(dim=b, device=device, dtype="float32"),
                nn.ReLU()
            )

        self.model = nn.Sequential(
            ConvBN(3,16,7,4),
            ConvBN(16,32,3,2),
            nn.Residual(nn.Sequential(
                ConvBN(32,32,3,1),
                ConvBN(32,32,3,1)
            )),
            ConvBN(32,64,3,2),
            ConvBN(64,128,3,2),
            nn.Residual(nn.Sequential(
                ConvBN(128,128,3,1),
                ConvBN(128,128,3,1)
            )),
            nn.Flatten(),
            nn.Linear(128, 128, device=self.device, dtype="float32"),
            nn.ReLU(),
            nn.Linear(128,10,device=self.device, dtype="float32")
        )
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.model(x)

        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION



class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, device=None, dtype="float32"):
        super(unetConv2, self).__init__()
        
        self.device = device
        self.dtype = dtype
        
        def ConvBN(a, b, k, s, device=self.device, dtype=self.dtype):
            return nn.Sequential(
                nn.Conv(a, b, k, s, device=device, dtype=dtype),
                nn.BatchNorm2d(dim=b, device=device, dtype=dtype),
                nn.ReLU()
            )

        if is_batchnorm:
            self.conv1 = ConvBN(in_size, out_size, 3, 1, device=self.device)
            self.conv2 = ConvBN(out_size, out_size, 3, 1, device=self.device)

        else:
            self.conv1 = nn.Sequential(
                nn.Conv(in_size, out_size, 3, 1, device=device, dtype=dtype), nn.ReLU())
            self.conv2 = nn.Sequential(
                nn.Conv(out_size, out_size, 3, 1, device=device, dtype=dtype), nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv=True, device=None, dtype="float32"):
        super(unetUp, self).__init__()
        
        self.conv = unetConv2(in_size, out_size, False, device=device, dtype=dtype)
        if is_deconv:
            self.up = nn.Conv_transposed(
                in_size, out_size, kernel_size=2, stride=2, device=device, dtype=dtype)
        else:
            pass
            # self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.shape[2] - inputs1.shape[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = ndl.ops.pad(inputs1, padding)
        return self.conv(ndl.ops.concat([outputs1, outputs2], 1))


class unet(nn.Module):
    def __init__(
        self, 
        feature_scale=4, 
        n_classes=21, 
        is_deconv=True, 
        in_channels=3, 
        is_batchnorm=False,
        device=None, 
        dtype="float32"
    ):
        super(unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm, device=device, dtype=dtype)
        self.maxpool1 = nn.Maxpool(kernel_size=2, device=device)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm, device=device, dtype=dtype)
        self.maxpool2 = nn.Maxpool(kernel_size=2, device=device)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm, device=device, dtype=dtype)
        self.maxpool3 = nn.Maxpool(kernel_size=2, device=device)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm, device=device, dtype=dtype)
        self.maxpool4 = nn.Maxpool(kernel_size=2, device=device)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm, device=device, dtype=dtype)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv, device=device, dtype=dtype)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv, device=device, dtype=dtype)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv, device=device, dtype=dtype)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv, device=device, dtype=dtype)

        # final conv (without any concat)
        self.final = nn.Conv(filters[0], n_classes, 1, device=device, dtype=dtype)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)