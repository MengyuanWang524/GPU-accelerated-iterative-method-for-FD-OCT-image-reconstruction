import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np  
import time
import torch.utils.data as Data
import pandas as pd
from numpy import fft, cos, inf, save, savetxt, zeros, array, exp
import matplotlib.pyplot as plt
from util import forward


    
def stochastic_gradient_descent(obj_est, target_fringe, Sk, matFourRe, lambdal1,
                                num_iters, loss, lr):
    train_losses= zeros([num_iters])
    train_l1loss = zeros([num_iters])
    train_mseloss= zeros([num_iters])
    fringe_est = zeros([])                      
    device = obj_est.device
    obj_est = torch.tensor(obj_est , requires_grad=True, device=device)
    # optimization variables and adam optimizer
    optvars = [{'params': obj_est}]
    optimizer = optim.Adam(optvars, lr=lr)
    gradr = torch.zeros( (512,num_iters) )
    start = time.time()
    scheduler = optim.lr_scheduler.StepLR(optimizer, 50, 0.25)
    tol = 1e-1

    # run the iterative algorithm
    
    for k in range(num_iters):

        optimizer.zero_grad()

        # forward propagation

        fringe_est = forward( obj_est, Sk, matFourRe )
  
        MSEloss = loss(fringe_est , target_fringe)

        L1loss = lambdal1* torch.mean( abs(obj_est) )

        lossValue = MSEloss + L1loss

        lossValue.backward()

        # Gardient Descent
        optimizer.step()

        # scheduler.step()

        if k > 1:
            train_losses[k] = MSEloss + L1loss
            train_l1loss[k] = L1loss
            train_mseloss[k] = MSEloss

        # if (k>100) & (abs(train_losses[k]-train_losses[k-1] )< tol):
        #     print(k)
        #     break

    end = time.time()
    print('totol time:', end-start)
    # torch.cuda.synchronize()
    return {'obj_est':obj_est,'loss':train_losses,'Sk':Sk, 'gradr':gradr,
            'fringe_est':fringe_est,'train_l1loss':train_l1loss,'train_mseloss':train_mseloss }

dtype = torch.float32
# Choose CPU or GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

fringe = pd.read_csv("fringe_onion.csv",header=None, dtype='float32')

fringe = torch.tensor(fringe.to_numpy()).to(device)

Sk = pd.read_csv("SkNo_onion.csv",header=None, dtype='float32')

Sk = torch.tensor(Sk.to_numpy()).to(device)
# Sk = Sk / torch.max(Sk)

rownum = fringe.size(dim = 0)

colnum = fringe.size(dim = 1)

M = rownum
N = rownum
factor = 2

T = 1000
### Decrease the fringe to 1/2 by seting deci to 2 
deci = 1 

deciST = int(1025 - M / deci / 2)

deciEND = int(1024 + M / deci / 2)

print(rownum)
print(colnum)

lambdal1 = 1000

lambda_st = 791.6e-9
lambda_end = 994e-9
# dz = 2e-6;
Dlambda = lambda_end - lambda_st

k_st = 2 * np.pi / lambda_end
k_end = 2 * np.pi / lambda_st
dk = (k_end - k_st) / N

dz_fft =  np.pi / (dk * N) 

## Factor = 2 half of the fft grid
dz = dz_fft /factor 


k = np.linspace(k_st, k_st + (N - 1) * dk, N )

gridRec = np.linspace(0, (T - 1) * dz, T ) 

X, Y = np.meshgrid(gridRec, k)

matFourRec = exp(- 2 * 1j * X * Y)

# # Optimization Initialization
step_size = 100

numIteration = 1000

lossFunc = nn.MSELoss().to(device)

time1 = time.time()

fftinit = abs(torch.fft.ifft(fringe, dim = 0,n =2048* factor))

time2 = time.time()

print('time2-time1:',time2-time1)

# FFT initialization

# r0 = fftinit[0:T,:]
# r0 = torch.tensor(r0, dtype=dtype, requires_grad=True).to(device)

# All zeros initialization
r0 = torch.zeros((T,colnum), dtype=dtype, requires_grad=True).to(device)

Sk_tensor = torch.tensor(Sk, dtype=dtype).to(device)


matFourRec_tensor = torch.tensor(matFourRec).to(device)

matCosRec = torch.tensor(matFourRec_tensor.real, dtype= dtype)

matSinRec = torch.tensor(matFourRec_tensor.imag, dtype= dtype)



SGD = stochastic_gradient_descent( r0, fringe, Sk_tensor, matCosRec,lambdal1, 
                            numIteration, loss=lossFunc, lr = step_size )


r_N = SGD['obj_est'] 

r_N = r_N.cpu().detach().numpy()  

wedgeRec = abs( r_N )

np.savetxt(fname = 'Onion_L1_%d_%f_%d_%d.csv'%(numIteration,step_size,factor,lambdal1), X = wedgeRec, delimiter=',')