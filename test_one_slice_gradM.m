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
source_folder = "Data/ct_scans/ct1";
target_folder = "/home/adwaye/matlab_projects/test_CT/Figures/ct1_experiment";

query     = strjoin([source_folder,"*slice_*"],'/');
filenames = dir(query);
nfiles    = size(filenames,1);

tempname = "curated2_pe_zslice_189.mat";%$change this to process a different slice
[filepath,fname,ext] = fileparts(tempname);


disp(tempname);
filename = strjoin([source_folder,tempname],"/");
name     = fname;




matfile = load(filename);
im_true = double(matfile.CT);
normA   = im_true-min(im_true(:));
normA   = normA./max(normA(:));
im_true = normA;
im_true = imresize(im_true,0.5,'Method','bilinear');
pad_up   = fix((512-size(im_true,1))/2);
pad_side = fix((512-size(im_true,2))/2);
im_true = padarray(im_true,[pad_up pad_side],0,'both');

mask = matfile.labels;
mask = imresize(mask,0.5,'bilinear');
mask = double(mask>0);
mask = padarray(mask,[pad_up pad_side],0,'both');

seed = 0 ;
SNR =@(x) 20 * log10(norm(im_true(:))/norm(im_true(:)-x(:)));

%%

[param_data.Ny,param_data.Nx] = size(im_true) ;
param_data.N = numel(im_true) ;
param_data.Nx = size(im_true,1);
param_data.Ny = size(im_true,2);

vol_geom = astra_create_vol_geom(param_data.Nx,param_data.Ny);
%create a parallel projection geometry with spacing =1 pixel, 384 
geom.spacing    = 1.34;% 1/(nx*ny);
geom.ndetectors = 384;
geom.n_angles   = 200;
geom.angles     = linspace2(0,pi,geom.n_angles);%use more angles
proj_geom       = astra_create_proj_geom('parallel', geom.spacing, geom.ndetectors, geom.angles);
geom.proj       = proj_geom;
geom.vol        = vol_geom;
stddev          = 1e-4;
W = opTomo('cuda',proj_geom,vol_geom);
W_scaled = W/W.normest;

num_angles = floor(1e-3*param_data.N) ;




phi = @(x) reshape(W_scaled*x(:),W.proj_size);

phit = @(v) reshape(W_scaled'*v(:),size(im_true)); 
norm_phi = op_norm(phi, phit, [param_data.Ny, param_data.Nx], 1e-4, 200, 0);  
Wscaled  = W/W.normest;
param_data.Phi  =phi;
param_data.Phit =phit;

param_data.normPhi = Wscaled.normest;%op_norm(param_data.Phi, param_data.Phit, [param_data.Ny, param_data.Nx], 1e-4, 200, 0);    
param_data.M       = size(param_data.Phi(im_true)) ;


one_sinogram  = ones(size(phi(im_true)));
geom_shape    = phit(one_sinogram);


figure(1), 
subplot 131, imagesc(im_true), axis image, colormap gray, colorbar, title("Ground Truth")
subplot 132, imagesc(mask), axis image, colormap gray, colorbar, title("pe locations")
subplot 133, imagesc(geom_shape), axis image, colormap gray, colorbar, title("Adjoint applied applied to one sinogram")



imshow(geom_shape,[]);colorbar();title("");

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
%subplot 132, imagesc(param_data.Mask), axis image, colormap gray, colorbar
subplot 133, imagesc(im_fbp), axis image, colormap gray, colorbar, xlabel(['FBP - SNR ', num2str(SNR(im_fbp))])

%%


[param_map.Psi, param_map.Psit] = wavelet_op(rand(size(im_true)), 4) ;
param_map.normPsi =1 ;
param_map.lambda = 1 ;
param_map.normPsi =1 ;
param_map.lambda = 1 ;


[Mop, Mopt] = gradient_op(rand(size(im_true))) ;
disp("Testing if the gradient forward op and the divergence are true adjoints")
xtmp  = rand(param_data.Ny,param_data.Nx) ;
ytmp  = Mop(rand(param_data.Ny,param_data.Nx)) ;% ytmp  = param_data.Phi(rand(param_data.Ny,param_data.Nx)) ;
Pxtmp = Mop(xtmp) ;% Pxtmp = param_data.Phi(xtmp) ;
Ptytmp = Mopt(ytmp) ;% Ptytmp = param_data.Phit(ytmp) ;
fwd = real(Pxtmp(:)'*ytmp(:)) ;
bwd = xtmp(:)'* Ptytmp(:) ;
disp('test adjoint operator')
disp(['fwd = ', num2str(fwd)])
disp(['bwd = ', num2str(bwd)])
disp(['diff = ', num2str(norm(fwd-bwd)/norm(fwd))])





