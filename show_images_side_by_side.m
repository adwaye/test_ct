clear all
clc
close all


addpath Algos/
addpath Data/
addpath Tools/
addpath(genpath('Tools'))

grad_norm_name = "L1"; %choose from L1, L2, Linf
M_norm_name    = "L2"; %choose from L1, L2, Linf
inpainting = false;%


algo_name = strjoin(["POCS",grad_norm_name,"gradM",M_norm_name,"Mx"],'_');
BUQO_algo = str2func(algo_name);yc