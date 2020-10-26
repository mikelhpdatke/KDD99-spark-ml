import pandas
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from time import time
import numpy as np


#nhan cac feature
col_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
kdd_data_10percent = pandas.read_csv("input_cnn.txt", header=None, names = col_names)
#print('du lieu input:')
#print (kdd_data_10percent.iloc[1,:])
#print (len(kdd_data_10percent.iloc[:,1]))
#print kdd_data_10percent.describe()
#print kdd_data_10percent['label'].value_counts()
num_features = [
    "duration","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
]
features = kdd_data_10percent[num_features].astype(float)
#print features.describe()
label_cnn =[]
label_cnn = kdd_data_10percent['label'].as_matrix()



# Training settings for pytorch
parser = argparse.ArgumentParser(description='PyTorch KDD99 Example')
parser.add_argument('--no_download_data', action='store_true', default=False,
                    help='Do not download data')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=8, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print("Number of epochs: ", args.epochs)
print("Batch size: ", args.batch_size)
print("Log interval: ", args.log_interval)
print("Learning rate: ", args.lr)
print(" Cuda: ", args.cuda)


#model CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv1d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout()
        self.fc1 = nn.Linear(140, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        print(x.size())
        # return x
        x = F.max_pool1d(x, 2)
        print(x.size())
        x = F.relu(x)
        print(x.size())
        x = self.conv2(x)
        print(x.size())
        x = self.conv2_drop(x)
        print(x.size())
        x = F.max_pool1d(x, 2)
        print(x.size())
        x = F.relu(x)
        print(x.size())
        x = x.view(-1, x.size(1) * x.size(2))
        print(x.size())
        x = F.relu(self.fc1(x))
        print(x.size())
        x = F.dropout(x, training=self.training)
        print(x.size())
        x = F.relu(self.fc2(x))
        print(x.size())
        x = F.log_softmax(x, dim=1)
        print(x.size())
        return x

# training process



# run with cuda if have
model = Net()
args.cuda = True
if args.cuda:
    model.cuda()

#optimizer model cnn
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

#convert dataframe to numpy data
input_model_data_numpy=kdd_data_10percent.values
input_model_data = torch.from_numpy(input_model_data_numpy).float().unsqueeze(1)

input_model_label_numpy = label_cnn
input_model_label = torch.from_numpy(input_model_label_numpy).float().unsqueeze(1)


# get batch
def get_batch(data):
	#print('do dai data '+str(len(data)/64))
	for i in range(0,len(data)/args.batch_size):
		tmp_data = data[i*args.batch_size: (i+1)*args.batch_size]
		yield tmp_data

# get batch cho label
def get_batch_label(data):
	#print('do dai data '+str(len(data)/64))
	for i in range(0,len(data)/args.batch_size):
		tmp_data = data[i*args.batch_size: (i+1)*args.batch_size]
		yield tmp_data

# target_1=label_cnn
# for i in range(args.batch_size):
# 	target_1 +=[4]
# target = np.asarray(target_1)
# target = torch.from_numpy(target)
# #print(target.size())
# #print(target)

# tao bo nhan cho torch train
batchs_label = get_batch_label(input_model_label)


# du lieu cho train
batchs = get_batch(input_model_data)
#print('he so batch')
losses=[]
idbatch =0
for batch_idx, batch in enumerate(batchs):
    #print(type(batch))
    idbatch+=1
    batch = batch.cuda()
    target = target.cuda()

    data, target = Variable(batch), Variable(target)
    optimizer.zero_grad()
    output = model(data)

    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    #print(loss.data)
    losses.append(loss.data[0])    
    print("\nTrain set batch {}: Average loss: {:.4f}".format(idbatch,sum(losses) / len(losses)))
# count acc training process
correct = 0
pred = output.data.max(1)[1].cpu().numpy()
label = target.data.cpu().numpy()
a=len(label)
b=(np.sum(pred == label))
print('Training completed!')
print(' acc training = {:.4f}'.format(float(b)/a))



#test process
model.eval()
test_loss = 0
correct = 0
#load data test
kdd_data_test = pandas.read_csv("test_input_cnn", header=None, names = col_names)



#convert dataframe to numpy data
input_model_data_numpy_test=kdd_data_test.values
input_model_data = torch.from_numpy(input_model_data_numpy_test).float().unsqueeze(1)


#label of test
target_1=[]
for i in range(len(input_model_data)):
    target_1 +=[4]
target_1[4]=7
target_1[10]=8
target_1[15]=7
target_1[20]=5
target = np.asarray(target_1)
target = torch.from_numpy(target)
#print(target.size())
#print(target)


input_model_data = input_model_data.cuda()
target = target.cuda()
data, target = Variable(input_model_data), Variable(target)
output = model(data)
loss = F.nll_loss(output, target)
#print("loss testing: "+str(loss.data[0]))
print('\nTest set: Average loss: {:.4f}\n'.format(loss.data[0]))

print('model: ')
print(model)
print('output model:')
pred = output.data.max(1)[1].cpu().numpy()
label = target.data.cpu().numpy()
print(pred)
print('output label:')
print(label)
a=len(label)
b=(np.sum(pred == label))
print(' acc = {:.4f}'.format(float(b)/a))
'''
for i in range(len(input_model_data)):
    pred = output.data.max(1)[1] # get the index of the max log-probability
    correct += pred.eq(target.data)
print(correct)
'''
