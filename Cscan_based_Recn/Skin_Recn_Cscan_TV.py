import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np  
import time
import torch.utils.data as Data
import pandas as pd
import h5py
from numpy import fft, cos, inf, save, savetxt, zeros, array, exp, conj, nan, isnan, pi, sin, seterr
import matplotlib.pyplot as plt
# import torch_tb_profiler as profiler

def forward(input_r, S_k, matFourRe):
    sampleRe =  torch.matmul(matFourRe, input_r)
    output_fringe = S_k *((sampleRe))
    return output_fringe

def TotalVariationLoss(input, weight):
    input_r = 10* torch.log10(abs(input))
    h_r = input_r.size(dim = 0)
    w_r = input_r.size(dim = 1)
    tv_h = torch.pow( input_r[1:,:] - input_r[:-1,:], 2 ).sum()
    tv_w = torch.pow( input_r[:,1:] - input_r[:,:-1], 2 ).sum()
    return weight*(tv_h+tv_w)/(h_r*w_r)

    
def stochastic_gradient_descent(obj_est, target_fringe, Sk, matFourRe,  weightTV,
                                num_iters, loss, lr, dev):
    train_losses = zeros([num_iters])
    train_tvloss = zeros([num_iters])
    train_mseloss = zeros([num_iters])
    # train_l1loss = zeros([num_iters])
    fringe_est = zeros([])                      
    device = dev
    obj_est = torch.tensor(obj_est , requires_grad=True, device=device)
    # optimization variables and adam optimizer
    optvars = [{'params': obj_est}]
    optimizer = optim.Adam(optvars, lr=lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer,50, 0.5)
    start = time.time()
    # run the iterative algorithm
    for k in range(num_iters):

        optimizer.zero_grad()

        # forward propagation
        fringe_est = forward( obj_est, Sk, matFourRe)

  
        MSEloss = loss(fringe_est , target_fringe)
        
        
        TVloss = TotalVariationLoss(obj_est, weightTV)

        lossValue = MSEloss + TVloss 


        lossValue.backward()

        # Gardient Descent
        optimizer.step()

        # scheduler.step()
        
        train_losses[k] = lossValue
        train_mseloss[k] = MSEloss
        train_tvloss[k] = TVloss
        # train_l1loss[k] = L1loss
    end = time.time()
    print('totol time:', end-start)

    return obj_est

dtype = torch.float32


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_ids = [0,1,2,3]
device_num = torch.cuda.device_count()

f = h5py.File("RawFringeface3D.h5", 'r+')
fringedataset = f['RawFringe']



fringe = zeros([100,4096,1000], dtype='float32') # for test


for index in range(len(fringe)):
    fringe[index,:,:] = fringedataset[index].T

# fringe = fringe[:,0:-1:2,:]
# fringe = fringe[:,0:-1:2,:]
fringe = torch.tensor(fringe).to(device)

Sk = pd.read_csv("Sk_skin.csv",header=None, dtype='float32')

Sk = torch.tensor(Sk.to_numpy()).to(device)


rownum = fringe.size(dim = 1)


colnum = fringe.size(dim = 2)


M = rownum
N = rownum
factor = 1
# T = int(N * factor / 2)
T = 600

print('bscan num:', len(fringe))

print(rownum)
print(colnum)


lambda_st = 1010e-9
lambda_end = 1095e-9
# dz = 2e-6;
Dlambda = lambda_end - lambda_st

k_st = 2 * np.pi / lambda_end

k_end = 2 * np.pi / lambda_st

dk = (k_end - k_st) / N

dz_fft =  np.pi / (dk * N) 

# dz = 1e-6

dz = dz_fft /factor

k = np.linspace(k_st, k_st + (N - 1) * dk, N )



gridRec = np.linspace(0, (T - 1) * dz, T ) 


X, Y = np.meshgrid(gridRec, k)

matFourRec = exp(- 2 * 1j * X * Y)


# # Optimization Initialization
# ref = 1
learning_rate = 0.00005

numIteration = 100  

lambdal1 = 0

weightTV = 0.1

lossFunc = nn.MSELoss().to(device)

fftinit = abs(torch.fft.ifft(fringe, dim = 1))

r0 = fftinit[:,0:T,:]

r0 = torch.tensor(r0, dtype=dtype, requires_grad=True).to(device)

# r0 = torch.zeros((T,colnum), dtype=dtype, requires_grad=True).to(device)
# Sk_No = Sk / (torch.max(Sk))
# Sk_No = Sk/ T * sum(Sk)
# Sk = torch.ones((M,1))
# Sk = Sk[0:-1:2,:]
# Sk = Sk[0:-1:2,:]
Sk_tensor = torch.tensor(Sk, dtype=dtype).to(device)

Sk_tensor2 = Sk_tensor.unsqueeze(0).repeat(device_num,1,1)

matFourRec_tensor = torch.tensor(matFourRec).to(device)

matCosRec = torch.tensor(matFourRec_tensor.real, dtype= dtype)

matSinRec = torch.tensor(matFourRec_tensor.imag, dtype= dtype)

matFourRec_tensor2 = matCosRec.unsqueeze(0).repeat(device_num,1,1)

class SGD(nn.Module):
    def __init__(self, numIteration = numIteration, lossFunc = nn.MSELoss().to(device),
                 learning_rate = learning_rate, weightTV = weightTV, dev = device): 

        super(SGD, self).__init__()
        # Setting parameters
        self.dev = dev
        self.lambdal1 = lambdal1
        self.weightTV = weightTV
        self.numIter = numIteration
        self.lr = learning_rate 
        self.loss = lossFunc

    def forward(self, r0, fringe_tensor, Sk_tensor, matFourRec_tensor):
        # print(self.r_0.shape)
        # Run algorithm
        obj_est= stochastic_gradient_descent( r0 ,fringe_tensor, Sk_tensor, matFourRec_tensor,
                                self.weightTV,self.numIter, self.loss, self.lr, self.dev )
        return   obj_est


model = SGD(  numIteration = numIteration, lossFunc = nn.MSELoss(),
                 learning_rate = learning_rate, weightTV = weightTV, dev = device )


if torch.cuda.device_count() > 1:
    print("Let's use", device_num, "GPUs")
    model = torch.nn.DataParallel(model, device_ids=device_ids,dim=0)
model.to(device)

obj_est = model(r0, fringe, Sk_tensor2, matFourRec_tensor2)

r_N = obj_est

r_N = r_N.cpu().detach().numpy()

wedgeRec = abs( r_N )

for index in range(1):
    np.savetxt(fname = 'Skin_SGD_para3D_%d_%d.csv'%(index,M), X = wedgeRec[index,:,:], delimiter=',')
