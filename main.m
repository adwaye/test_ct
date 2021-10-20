% Test UQ for MRI
% discrete Fourier measurements + wavelet
%% ------------------------------------------------------------------------

clear all
clc
close all


addpath Algos/
addpath Data/
addpath Tools/
addpath(genpath('Tools'))


%% 

load('phantomMRI.mat')
seed = 0 ;
SNR =@(x) 20 * log10(norm(im_true(:))/norm(im_true(:)-x(:)));

%%

[param_data.Ny,param_data.Nx] = size(im_true) ;
param_data.N = numel(im_true) ;
param_data.Nx = size(im_true,1);
param_data.Ny = size(im_true,2);

vol_geom = astra_create_vol_geom(param_data.Nx,param_data.Ny);
%create a parallel projection geometry with spacing =1 pixel, 384 
geom.spacing    = 1.0;% 1/(nx*ny);
geom.ndetectors = 384;
geom.angles     = linspace2(0,pi,50);%use more angles
proj_geom       = astra_create_proj_geom('parallel', geom.spacing, geom.ndetectors, geom.angles);
geom.proj       = proj_geom;
geom.vol        = vol_geom;
stddev          = 1e-4;

num_angles = floor(1e-3*param_data.N) ;
%[Mask_fft,param_data.Mask] = radial_sampling_op(im_true,num_angles);
param_data.Mask = circular_mask(param_data.Nx,param_data.Ny);

% param_data.Phi =@(x) TF(x,param_data.N) ;
% param_data.Phit =@(y) TFadj(y,param_data.N) ;

param_data.Phi  =@(x) fwd_op(x,geom,param_data.Mask) ;
param_data.Phit =@(y) bwd_op(y,geom,param_data.Mask) ;

param_data.normPhi = op_norm(param_data.Phi, param_data.Phit, [param_data.Ny, param_data.Nx], 1e-4, 200, 0);    
param_data.M       = size(param_data.Phi(im_true)) ;

% -------------------------------------------------
% test adjoint operator
xtmp  = rand(param_data.Ny,param_data.Nx) ;
ytmp  = param_data.Phi(rand(param_data.Ny,param_data.Nx)) ;
Pxtmp = param_data.Phi(xtmp) ;
Ptytmp = param_data.Phit(ytmp) ;
fwd = real(Pxtmp(:)'*ytmp(:)) ;
bwd = xtmp(:)'* Ptytmp(:) ;
disp('test adjoint operator')
disp(['fwd = ', num2str(fwd)])
disp(['bwd = ', num2str(bwd)])
disp(['diff = ', num2str(norm(fwd-bwd)/norm(fwd))])
% -------------------------------------------------

%%
param_data.sig_noise = 5e-4 ;
param_data.y0 = param_data.Phi(im_true) ;
rng(seed);
noise = (randn(param_data.M) ) * param_data.sig_noise/sqrt(2) ; %use gaussian for starters should be poisson as poisson is more accurate
param_data.y = param_data.y0 + noise ;

param_data.data_eps = sqrt(2*prod(param_data.M) + 2* sqrt(4*prod(param_data.M))) *  param_data.sig_noise ;

% FBP estimate
im_fbp = param_data.Phit(param_data.y) ;
figure, 
subplot 131, imagesc(im_true), axis image, colormap gray, colorbar
subplot 132, imagesc(param_data.Mask), axis image, colormap gray, colorbar
subplot 133, imagesc(im_fbp), axis image, colormap gray, colorbar, xlabel(['FBP - SNR ', num2str(SNR(im_fbp))])

%%

[param_map.Psi, param_map.Psit] = wavelet_op(rand(size(im_true)), 4) ;
param_map.normPsi =1 ;
param_map.lambda = 1 ;

sig1 = 0.7/param_map.normPsi ;
sig2 = 0.7/param_data.normPhi ;
tau = 0.99 / (sig1*param_map.normPsi + sig2*param_data.normPhi) ;
disp(['tau = ', num2str(tau)])
disp(['sig1 = ', num2str(sig1)])
disp(['sig2 = ', num2str(sig2)])

max_it = 16000 ;
stop_it = 1e-4 ;
stop_norm = 1e-4 ;

[xmap, fid, reg, norm_it, snr_it] = MAP_primal_dual(param_data, param_map, tau, sig1, sig2, max_it, stop_it, stop_norm, SNR) ;

figure, 
subplot 221, imagesc(im_true), axis image, colormap gray, colorbar, xlabel('true')
subplot 222, imagesc(param_data.Mask), axis image, colormap gray, colorbar, xlabel('mask')
subplot 223, imagesc(im_fbp), axis image, colormap gray, colorbar, xlabel(['FBP - SNR ', num2str(SNR(im_fbp))])
subplot 224, imagesc(xmap), axis image, colormap gray, colorbar, xlabel(['xmap - SNR ', num2str(snr_it(end))])




%%

param_hpd.lambda_t = param_data.N / sum(abs(param_map.Psit(xmap))) ;


alpha = 1e-2 ; 
talpha = sqrt( (16*log(3/alpha)) / param_data.N );
HPDconstraint = param_hpd.lambda_t* sum(abs(param_map.Psit(xmap))) ...
                + param_data.N*(1+talpha);
param_hpd.HPDconstraint = HPDconstraint/param_hpd.lambda_t ;


param_hpd.Psit = param_map.Psit ;
param_hpd.Psi = param_map.Psi ;
param_hpd.normPsi = param_map.normPsi ;
param_hpd.lambda = param_map.lambda ;

%%
cropx = 157:165 ;
cropy = 128:140 ;
mincrop = 0.25 ;
maxcrop = 0.32 ;

mask_struct = zeros(size(xmap)) ;
mask_struct(cropy,cropx) = 1 ;
mask_struct = mask_struct.*(xmap>mincrop) ;
mask_struct = mask_struct.*(xmap<maxcrop) ;
mask_struct(128, 160:169) = 0 ;
mask_struct(131, 157:158) = 0 ;
mask_struct(137:139, 163:165) = 0 ;
mask_struct = imdilate(mask_struct,strel('disk',1));


tmp = xmap ; tmp(mask_struct>0) = 0 ;
figure, 
subplot 221, imagesc(mask_struct), axis image, colormap gray
subplot 222, imagesc(tmp), axis image, colormap gray
subplot 223, imagesc(im_true), axis image, colormap gray
subplot 224, imagesc(xmap), axis image, colormap gray