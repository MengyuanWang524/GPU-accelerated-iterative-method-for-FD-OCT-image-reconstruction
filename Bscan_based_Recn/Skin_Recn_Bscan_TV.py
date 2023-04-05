import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np  
import time
import torch.utils.data as Data
import pandas as pd
from numpy import fft, cos, inf, save, savetxt, zeros, array, exp, conj, nan, isnan, pi, sin, seterr
import matplotlib.pyplot as plt
import torch_tb_profiler as profiler
from util import forward,TotalVariationLoss

    
def stochastic_gradient_descent(obj_est, target_fringe, Sk, matFourRe, weightTV, lambdal1,
                                num_iters, loss, lr):
    train_losses = zeros([num_iters])

    train_tvloss = zeros([num_iters])

    train_mseloss = zeros([num_iters])

    train_l1loss = zeros([num_iters])

    fringe_est = zeros([])       

    device = obj_est.device
    
    obj_est = torch.tensor(obj_est , requires_grad=True, device=device)
    # optimization variables and adam optimizer
    optvars = [{'params': obj_est}]

    optimizer = optim.Adam(optvars, lr=lr)

    tol = 1e-4
    scheduler = optim.lr_scheduler.StepLR(optimizer,50, 0.5)

    start = time.time()
    # run the iterative algorithm
    for k in range(num_iters):

        optimizer.zero_grad()

        # forward propagation

        fringe_est = forward( obj_est, Sk, matFourRe, ref )
  
        MSEloss = loss(fringe_est , target_fringe)
        
        L1loss = lambdal1* torch.mean( abs(obj_est) )

        TVloss = TotalVariationLoss(obj_est, weightTV)

        lossValue = MSEloss + TVloss + L1loss

        lossValue.backward()

        # Gardient Descent
        optimizer.step()

        # scheduler.step()
        
        train_losses[k] = lossValue

        train_mseloss[k] = MSEloss

        train_tvloss[k] = TVloss

        train_l1loss[k] = L1loss

        # if (k>500) & (abs(train_losses[k]-train_losses[k-1] )< tol):
        #     print(k)
        #     break

    end = time.time()
    print('totol time:', end-start)

    return {'obj_est':obj_est}

dtype = torch.float32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fringe = pd.read_csv("fringe_skin.csv",header=None, dtype='float32')

fringe = torch.tensor(fringe.to_numpy()).to(device)

Sk = pd.read_csv("Sk_skin.csv",header=None, dtype='float32')

Sk = torch.tensor(Sk.to_numpy()).to(device)

rownum = fringe.size(dim = 0)

colnum = fringe.size(dim = 1)

M = rownum

N = rownum

factor = 1

T = 600

print(rownum)
print(colnum)

lambdal1 = 0

weightTV = 0.01

lambda_st = 1010e-9

lambda_end = 1095e-9


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
ref = 1
step_size = 0.00005

numIteration = 500

lossFunc = nn.MSELoss().to(device )

fftinit = abs(torch.fft.ifft(fringe, dim = 0))

r0 = fftinit[0:T,:]

r0 = torch.tensor(r0, dtype=dtype, requires_grad=True).to(device)

# r0 = torch.zeros((T,colnum), dtype=dtype, requires_grad=True).to(device)
# Sk_No = Sk / (torch.max(Sk))
# Sk_No = Sk/ T * sum(Sk)
# Sk = torch.ones((M,1))

Sk_tensor = torch.tensor(Sk, dtype=dtype).to(device)


matFourRec_tensor = torch.tensor(matFourRec).to(device)

matCosRec = torch.tensor(matFourRec_tensor.real, dtype= dtype)

matSinRec = torch.tensor(matFourRec_tensor.imag, dtype= dtype)


SGD = stochastic_gradient_descent( r0, fringe, Sk_tensor, matCosRec,weightTV, lambdal1,
                            numIteration, loss=lossFunc, lr = step_size )

r_N = SGD['obj_est'] 

r_N = r_N.cpu().detach().numpy() 

Rec =  abs(r_N) 

np.savetxt(fname = 'Skin_%d_%f_%d_%f_%f.csv'%(numIteration, step_size, factor, weightTV,lambdal1), X = Rec, delimiter=',' )            
