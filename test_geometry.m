
clear all
%clc
close all


addpath Algos/
addpath Data/
addpath Tools/
addpath(genpath('Tools'))
% P = phantom(256);
filename = "Data/ct_scans/ct1";
load('Data/ct_scans/ct1/artcon_zslice_297.mat')
P = single(CT);
%load("/home/adwaye/PycharmProjects/CT-UQ/CTimages.mat");
%P = gtruth*5;
nx=size(P,1);
ny=size(P,2);


%we will create a linear opeator that will create a sinogram and find its
%adjoint
vol_geom = astra_create_vol_geom(nx,ny);
%create a parallel projection geometry with spacing =1 pixel, 384 
geom.spacing = 1.35;
geom.ndetectors = 384;
geom.angles     =  linspace2(0,pi,200);
proj_geom = astra_create_proj_geom('parallel', geom.spacing, geom.ndetectors, geom.angles);
geom.proj  = proj_geom;
geom.vol = vol_geom;
stddev   = 1e-4;

W = opTomo('cuda',proj_geom,vol_geom);



%y = CTbeam(P,geom.proj,geom.vol);

%phi = @(x) CTbeam(x,geom,mask) ;
%phi = @(x) reshape(W*x(:),[W.proj_size]);
phi = @(x) reshape(W*x(:),W.proj_size);
%astra_fun = @(v,T) astra_wrap(v,T,vol_geom,proj_geom);
%phit = @(x) lsqr(@astra_fun,reshape(x',[numel(x) 1]),1e-4,500);
phit = @(v) reshape(W'*v(:),size(P)); % can package this into an opera




%volume = astra_create_backprojection_cuda(y, proj_geom, vol_geom);

%y = phi(P.*mask);
y = phi(P);
recon = phit(y);
%recon = CTbeamAdj(y,geom.proj,geom.vol);
y_scaled = rescale(y);
Mx = size(y,1);My = size(y,2);
noise = (randn(Mx,My)) * stddev/sqrt(2);
noisy_data = y+noise;
%Noise = poissrnd(I1);
%recon_noisy = phit(noisy_data);
figure(1); imshow(P, []);colorbar();title("Phantom");
figure(2); imshow(y, []);colorbar();title("sinogram");
figure(3);imshow(recon,[]);colorbar();title("adjoint of phi");
figure(4);imshow(noisy_data,[]);colorbar();title("noisy data");
%figure(5);imshow(recon_noisy,[]);colorbar();title("Reconstructed from noisy data");




one_sinogram  = ones(size(noisy_data));
geom_shape = phit(one_sinogram);
figure(5);imshow(geom_shape,[]);colorbar();title("Adjoint applied to sinogram containing only ones");




