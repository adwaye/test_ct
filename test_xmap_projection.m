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

grad_norm_name = "L2"; %choose from L1, L2, Linf
M_norm_name    = "L2"; %choose from L1, L2, Linf
inpainting = false;%
use_dil_mask= false;
plot_all = true;

grad_scheme_name = "";%bw_fw or ""
algo_name = strjoin(["POCS",grad_norm_name,"gradM",M_norm_name,"Mx"],'_');
BUQO_algo = str2func(algo_name);
use_full_size = false;


source_folder = "Data/ct_scans/ct1";
if grad_scheme_name ~= ""
    grad_op_name = strjoin(["gradient_op",grad_scheme_name],"_");
    folder_name   = strjoin(["ct1_experiment",grad_norm_name,"gradM",grad_scheme_name,M_norm_name,"M"],'_');
else
    grad_op_name = "gradient_op";
    folder_name   = strjoin(["ct1_experiment",grad_norm_name,"gradM",M_norm_name,"M"],'_');
end

if use_full_size
    folder_name = strjoin(["full_size",folder_name],'_');
end
if use_dil_mask
    folder_name = strjoin([folder_name,"dil_mask"],'_');
end
grad_op   = str2func(grad_op_name);
BUQO_algo = str2func(algo_name);

%% 


target_folder = strjoin(["/home/adwaye/matlab_projects/test_CT/Figures",folder_name],'/');
mkdir(target_folder);
mkdir(strjoin([target_folder,"png"],'/'))

query     = strjoin([source_folder,"*slice_*"],'/');
filenames = dir(query);
nfiles    = size(filenames,1);

% tempname = "curated2_pe_zslice_189.mat";%$change this to process a different slice
%tempname = "curated2_pe_xslice_225.mat";
tempname = "curated2_pe_yslice_266.mat";
[filepath,fname,ext] = fileparts(tempname);


disp(tempname);
filename = strjoin([source_folder,tempname],"/");
name     = fname;




matfile = load(filename);
im_true = double(matfile.CT);
normA   = im_true-min(im_true(:));
normA   = normA./max(normA(:));
im_true = normA;
mask    = matfile.labels;
if ~use_full_size
    im_true = imresize(im_true,0.5,'Method','bilinear');
    mask = imresize(mask,0.5,'bilinear');
end
view_size = 1.05*sqrt(size(im_true,1)^2+size(im_true,2)^2);
max_size = 350;%2*max(size(im_true));
pad_up   = fix((max_size-size(im_true,1))/2);
pad_side = fix((max_size-size(im_true,2))/2);
im_true = padarray(im_true,[pad_up pad_side],0,'both');
if size(im_true,1) ~= max_size
    im_true = padarray(im_true,[1 0],0,'post');
end
if size(im_true,2) ~= max_size
    im_true = padarray(im_true,[0 1],0,'post');
end



mask = double(mask>0);
mask = padarray(mask,[pad_up pad_side],0,'both');
if size(mask,1) ~= max_size
    mask = padarray(mask,[1 0],0,'post');
end
if size(mask,2) ~= max_size
    mask= padarray(mask,[0 1],0,'post');
end
% mask = imdilate(mask,strel('disk',2));
if use_dil_mask
    mask(182:184,181)=1;
    mask(181:182,182)=1;
    mask(180,183)=1;
    mask(179,184)=1;
    mask(178,185:189)=1;
    
end

seed = 0 ;
SNR =@(x) 20 * log10(norm(im_true(:))/norm(im_true(:)-x(:)));

%%
% im_true = zeros(350,350);
[param_data.Ny,param_data.Nx] = size(im_true) ;
param_data.N = numel(im_true) ;
param_data.Nx = size(im_true,1);
param_data.Ny = size(im_true,2);


detector_setup = [450,450,450,450];
angle_setup    = [450,300,200,100];
noise_array    = [0.00005,0.0002,0.0003,0.001];
alpha_array  = [0.01];
n_setup = size(angle_setup,2);
n_alpha = max(size(alpha_array));
n_noise = max(size(noise_array));
noise_index = 1;
alpha_index = 1;
index_setup = 1;

geom.n_angles   = angle_setup(index_setup);%900;
geom.ndetectors = detector_setup(index_setup);%900;
vol_geom = astra_create_vol_geom(param_data.Nx,param_data.Ny);
geom.spacing    = view_size/geom.ndetectors;%1/sqrt(2);% 1/(nx*ny);
geom.angles     = linspace2(0,pi,geom.n_angles);%use more angles
proj_geom       = astra_create_proj_geom('parallel', geom.spacing, geom.ndetectors, geom.angles);
geom.proj       = proj_geom;
geom.vol        = vol_geom;
stddev          = 1e-4;
W = opTomo('cuda',proj_geom,vol_geom);
W_scaled = W/W.normest;

%num_angles = floor(1e-3*param_data.N) ;
phi      = @(x) reshape(W_scaled*x(:),W.proj_size);
phit     = @(v) reshape(W_scaled'*v(:),size(im_true)); 
norm_phi = op_norm(phi, phit, [param_data.Ny, param_data.Nx], 1e-4, 200, 0);  
Wscaled  = W/W.normest;
param_data.Phi  = phi;
param_data.Phit = phit;

param_data.normPhi = Wscaled.normest;%op_norm(param_data.Phi, param_data.Phit, [param_data.Ny, param_data.Nx], 1e-4, 200, 0);    
param_data.M       = size(param_data.Phi(im_true)) ;


one_sinogram  = ones(size(phi(im_true)));
geom_shape    = phit(one_sinogram);

% mask_copy = mask;
if plot_all
    figure(1), 
    subplot 131, imagesc(im_true), axis image, colormap gray, colorbar, title("Ground Truth")%,ax = gca, ax.YLim = [lower_x upper_x], ax.XLim = [lower_y upper_y];
    subplot 132, imagesc(mask), axis image, colormap gray, colorbar, title("pe locations")%,ax = gca, ax.YLim = [lower_x upper_x], ax.XLim = [lower_y upper_y];
    % subplot 133, imagesc((1-mask_copy).*im_true), axis image, colormap gray, colorbar, title("Masked area"),ax = gca, ax.YLim = [lower_x upper_x], ax.XLim = [lower_y upper_y];
    subplot 133, imagesc(geom_shape), axis image, colormap gray, colorbar, title("Adjoint applied applied to one sinogram")


    imshow(geom_shape,[]);colorbar();title("");
end
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
param_data.sig_noise = noise_array(noise_index) ;
param_data.y0 = param_data.Phi(im_true) ;
rng(seed);
noise = (randn(param_data.M) ) * param_data.sig_noise/sqrt(2) ; %use gaussian for starters should be poisson as poisson is more accurate
param_data.y = param_data.y0 + noise ;

param_data.data_eps = sqrt(2*prod(param_data.M) + 2* sqrt(4*prod(param_data.M))) *  param_data.sig_noise ;

% FBP estimate
im_fbp = param_data.Phit(param_data.y) ;
if plot_all
    figure, 
    subplot 131, imagesc(im_true), axis image, colormap gray, colorbar
%subplot 132, imagesc(param_data.Mask), axis image, colormap gray, colorbar
    subplot 133, imagesc(im_fbp), axis image, colormap gray, colorbar, xlabel(['FBP - SNR ', num2str(SNR(im_fbp))])
end

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

max_it = 20000 ;
stop_it = 1e-4 ;
stop_norm = 1e-4 ;




forward_param_names       = strjoin([param_data.sig_noise,"noise",geom.ndetectors,"ndtct",geom.n_angles,"agls",geom.spacing,"grdsz"],"_");
results_name        = strjoin([name,"forward_problem_results",forward_param_names,"mat"],["_","_","."]);
results_path        = strjoin([target_folder ,results_name],'/');
if isfile(results_path)
    load(results_path)
else
    [xmap, fid, reg, norm_it, snr_it, time_it,time_total] = MAP_primal_dual(param_data, param_map, tau, sig1, sig2, max_it, stop_it, stop_norm, SNR) ;    
    save(results_path,'xmap','fid','reg','norm_it','snr_it','time_it','time_total')
end
xmap = double(xmap);
% 





if plot_all
    fig = figure, 
    subplot 221, imagesc(im_true), axis image, colormap gray, colorbar, xlabel('true')
    subplot 222, imagesc(mask), axis image, colormap gray, colorbar, xlabel('mask')
    subplot 223, imagesc(im_fbp), axis image, colormap gray, colorbar, xlabel(['FBP - SNR ', num2str(SNR(im_fbp))])
    subplot 224, imagesc(xmap), axis image, colormap gray, colorbar, xlabel(['xmap - SNR ', num2str(snr_it(end))])
end



%%
mask_struct = mask;
[row_mask col_mask] = find(mask);
max_y = max(col_mask,[],'all');
max_x = max(row_mask,[],'all');
min_y = max(col_mask,[],'all');
min_x = max(row_mask,[],'all');
upper_x = max_x+15;
lower_x =  min_x-15;
upper_y = max_y+15;
lower_y = min_y-15;
tmp = xmap ; tmp(mask_struct>0) = 0 ;

fig = figure(99), 
subplot 221, imagesc(mask),colorbar(),title("mask"), axis image, colormap gray
subplot 222, imagesc(tmp),colorbar(),title("reverse masked image"), axis image, colormap gray
subplot 223, imagesc(im_true),colorbar(),title("ground truth"), axis image, colormap gray
subplot 224, imagesc(xmap),colorbar(),title("map estimate"), axis image, colormap gray
fig_name = strjoin([name,"fwd_res",forward_param_names,"fig"],["_","_","."]);
fig_path = strjoin([target_folder,fig_name],"/");

close(fig)


%imwrite(im_true,'figures/g_truth')



%===========================================================================
% RUNNING BUQO NOW
%===========================================================================


param_struct.Mask       = mask_struct; % Mask to select structure to test
param_struct.choice_set = 'l2_const' ;



if strcmp(tempname,'curated2_pe_zslice_189.mat')
    texture_mask                  = zeros(size(xmap));
    texture_mask(255:262,250:256) = 1;
    texture_mask(260:275,228:244) = 1;
else
    texture_mask = matfile.texture_mask;
    texture_mask = imresize(texture_mask,0.5,'Method','nearest');
    texture_mask= padarray(texture_mask,[pad_up pad_side],0,'both');
    if size(texture_mask,1) ~= max_size
        texture_mask = padarray(texture_mask,[1 0],0,'post');
    end
    if size(texture_mask,2) ~= max_size
        texture_mask = padarray(texture_mask,[0 1],0,'post');
    end
end


[fx, fy] = gradient(xmap);
sampled_gradients = [fx(texture_mask>0),fy(texture_mask>0)];

mean_values_grad  = [0,0,0,0,0];
quantiles    = [0.6,0.7,0.9];

bound_num = 1;

sampled_pixels = xmap(texture_mask>0);

mean_values_M  = [median(sampled_pixels(:)),median(sampled_pixels(:)),median(sampled_pixels(:)),median(sampled_pixels(:)),median(sampled_pixels(:))];


param_struct.l2_mean_pix = median(sampled_pixels(:));
pix_quantile = quantiles(bound_num);
param_struct.l2_bound_pix = max([quantile(sampled_pixels(:),pix_quantile)-median(sampled_pixels(:)),median(sampled_pixels(:))-quantile(sampled_pixels(:),1-pix_quantile)]);

param_struct.l2_mean_grad = 0.0;
grad_quantile = quantiles(bound_num);
param_struct.l2_bound_grad= max([quantile(sampled_gradients(:),grad_quantile)-median(sampled_gradients(:)),median(sampled_gradients(:))-quantile(sampled_gradients(:),1-grad_quantile)]);


[Gradop, Gradopt] = grad_op(rand(size(im_true)),mask) ;

Mask_op = sparse(sum(param_struct.Mask(:)), numel(param_struct.Mask)) ;
Mask_op_comp = sparse(numel(param_struct.Mask)-sum(param_struct.Mask(:)), numel(param_struct.Mask)) ;
i = 1; ic = 1;
for n = 1:numel(param_struct.Mask)
    if(param_struct.Mask(n))==0
        Mask_op_comp(ic,n) = 1 ;
        ic = ic+1 ;
    else
        Mask_op(i,n) = 1;
        i = i+1 ;
    end
end


param_struct.Gradop = Gradop ;
param_struct.Gradopt = Gradopt;
param_struct.Mask_op = Mask_op;

param_algo.NbIt = 10000 ;
param_algo.stop_dist = 1e-4 ;
param_algo.stop_norm2 = 1e-4 ;
param_algo.stop_norm1 = 1e-4 ;
param_algo.stop_err_smooth = 1e-4 ;


proj_S = Project_to_S(xmap, param_algo, param_data,  param_struct);










