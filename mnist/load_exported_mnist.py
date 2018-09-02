from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import tensorflow as tf
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')

parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')

parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = torch.load("./saved_model/exported_mnist")

# params = list(model.named_parameters())
# print(len(params))
# for i in range(len(params)):
#     name, param = params[i]
#     print("[%s] %s" % (name, param.size()))
# for i, c in enumerate(model.children()):
# 	print(i, c)

def test():
    model.eval()
    input = torch.ones(1, 1, 28, 28)
    if args.cuda:
    	input = input.cuda()
    input = Variable(input, volatile=True)
    out = model(input)
    print(out)

test()

def conv2d(c, x):
    padding = 'VALID' if c.padding[0] is 0 else 'SAME'
    filters = c.out_channels
    size = c.kernel_size
    parameters = [p for p in c.parameters()]
    W = parameters[0].data.cpu().numpy()
    if len(parameters) > 1:
        b = parameters[1].data.cpu().numpy()

    W = np.transpose(W,[2,3,1,0])

    wi = tf.constant_initializer(W)
    if len(parameters) > 1:
        bi = tf.constant_initializer(b)

    Wt = tf.get_variable('weights', shape= W.shape, initializer = wi)
    if len(parameters) > 1:
        bt = tf.get_variable('bias', shape = b.shape, initializer = bi)

    # print(x.shape)
    # print(Wt.shape)

    x = tf.nn.conv2d(x , Wt, [1, c.stride[0], c.stride[1], 1], padding)
    if len(parameters) > 1:
        x = tf.nn.bias_add(x, bt)

    x = tf.nn.max_pool(x, [1, c.kernel_size[0], c.kernel_size[1], 1], strides = [1, c.stride[0], c.stride[1], 1], padding = padding)

    x = tf.nn.relu(x)

    return x

def dropout(c, x):
    return tf.reshape(x, [-1, 320])

def linear(c, x):
    parameters = [p for p in c.parameters()]
    W = parameters[0].data.cpu().numpy()
    if len(parameters) > 1:
        b = parameters[1].data.cpu().numpy()

    # print(W.shape)
    # print(b.shape)
    W = np.transpose(W)

    wi = tf.constant_initializer(W)
    if len(parameters) > 1:
        bi = tf.constant_initializer(b)

    Wt = tf.get_variable('weights', shape= W.shape, initializer = wi)
    if len(parameters) > 1:
        bt = tf.get_variable('bias', shape = b.shape, initializer = bi)

    x = tf.matmul(x, Wt)
    if len(parameters) > 1:
        x = tf.add(x, bt)

    x = tf.nn.relu(x)

    return x

type_lookups = {}

type_lookups[torch.nn.modules.conv.Conv2d] = conv2d
type_lookups[torch.nn.modules.dropout.Dropout2d] = dropout
type_lookups[torch.nn.modules.linear.Linear] = linear




tf.reset_default_graph()
input_image = tf.placeholder('float', shape = [1, 28, 28, 1], name = 'input_image')

x = input_image

for idx,c in enumerate(model.children()):
    # print(c.__class__)
    c_class = c.__class__
    if c_class in type_lookups:
        with tf.variable_scope('layer' + str(idx)):
            x = type_lookups[c_class](c, x)

features = x
classifier = tf.nn.log_softmax(features)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
scores = sess.run(classifier, feed_dict = {input_image: np.ones([1, 28, 28, 1])})
print(scores)