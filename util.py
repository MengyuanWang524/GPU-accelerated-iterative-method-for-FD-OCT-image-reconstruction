import torch
import torch.nn as nn

def forward(input_r, S_k, matFourRe, ref):
    sampleRe =  torch.matmul(matFourRe, input_r)
    # sampleIm =  torch.matmul(matFourIm, input_r)
    output_fringe = S_k *((sampleRe))
    return output_fringe

def TotalVariationLoss(input_r, weight):
    # input_r = 10* torch.log10( abs(input_r) )
    h_r = input_r.size(dim = 0)
    w_r = input_r.size(dim = 1)
    tv_h = torch.pow( input_r[1:,:] - input_r[:-1,:], 2 ).sum()
    tv_w = torch.pow( input_r[:,1:] - input_r[:,:-1], 2 ).sum()
    return weight*(tv_h+tv_w)/(h_r*w_r)