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
target_folder = "/home/adwaye/matlab_projects/test_CT/Figures/inpainting_experiments";

query     = strjoin([source_folder,"*slice_*"],'/');
filenames = dir(query);
nfiles    = size(filenames,1);

tempname = "curated2_pe_xslice_225.mat";%$change this to process a different slice
% tempname = "curated2_pe_zslice_189.mat";%$change this to process a different slice
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
if size(im_true,1) ~= 512
    im_true = padarray(im_true,[1 0],0,'post');
end
if size(im_true,2) ~= 512
    im_true = padarray(im_true,[0 1],0,'post');
end

mask = matfile.labels;
mask = imresize(mask,0.5,'bilinear');
mask = double(mask>0);
mask = padarray(mask,[pad_up pad_side],0,'both');
if size(mask,1) ~= 512
    mask = padarray(mask,[1 0],0,'post');
end
if size(mask,2) ~= 512
    mask= padarray(mask,[0 1],0,'post');
end

[row_mask col_mask] = find(mask);
max_y = max(col_mask,[],'all');
max_x = max(row_mask,[],'all');
min_y = max(col_mask,[],'all');
min_x = max(row_mask,[],'all');
upper_x = max_x+15;
lower_x =  min_x-15;
upper_y = max_y+15;
lower_y = min_y-15;
% tmp = xmap ; tmp(mask_struct>0) = 0 ;
% 
% seed = 0 ;
% SNR =@(x) 20 * log10(norm(im_true(:))/norm(im_true(:)-x(:)));

%%

[param_data.Ny,param_data.Nx] = size(im_true) ;
param_data.N = numel(im_true) ;
param_data.Nx = size(im_true,1);
param_data.Ny = size(im_true,2);

vol_geom = astra_create_vol_geom(param_data.Nx,param_data.Ny);
%create a parallel projection geometry with spacing =1 pixel, 384 
geom.spacing    = 1.34;% 1/(nx*ny);
geom.ndetectors = 500;
geom.n_angles   = 400;
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
seed = 0;
rng(seed);
noise = (randn(param_data.M) ) * param_data.sig_noise/sqrt(2) ; %use gaussian for starters should be poisson as poisson is more accurate
param_data.y = param_data.y0 + noise ;

param_data.data_eps = sqrt(2*prod(param_data.M) + 2* sqrt(4*prod(param_data.M))) *  param_data.sig_noise ;



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
    disp("forward problem previously simulated; loading results")
    load(results_path)
else
    disp("solving forward problems")
    [xmap, fid, reg, norm_it, snr_it, time_it,time_total] = MAP_primal_dual(param_data, param_map, tau, sig1, sig2, max_it, stop_it, stop_norm, SNR) ;    
    save(results_path,'xmap','fid','reg','norm_it','snr_it','time_it','time_total')
end
xmap = double(xmap);
%%





mask_struct = mask;
param_struct.Mask       = mask_struct; % Mask to select structure to test
param_struct.choice_set = 'l2_const' ;



kernels = {[3],[3,5],[5],[5,7],[7],[7,9],[3,5,7],[3,7,9],[5,9],[9],[11]};

for i=1:size(kernels,2)
    param_struct.Size_Gauss_kern = cell2mat(kernels(i)) ;
    disp('create inpainting operator...')
    disp('     -- recursive - can take few minutes... --')
    param_struct.L = create_inpainting_operator_test(param_struct.Mask, param_struct.Size_Gauss_kern, xmap) ;
    disp('...done')
    param_struct.L = sparse(param_struct.L) ;




    %%
    



    %%




    xmap_S = xmap ;
    xmap_S(param_struct.Mask>0) = param_struct.L*xmap(param_struct.Mask==0);


    im_true_S = im_true;
    im_true_S(param_struct.Mask>0) = param_struct.L*im_true_S(param_struct.Mask==0);

    fig = figure(99), 
    pause(0.1);
    frame_h = get(handle(gcf),'JavaFrame');
    set(frame_h,'Maximized',1)
    subplot 231, imagesc(im_true.*mask),colorbar(),title("Masked Ground Truth"), axis image, colormap gray, ax = gca, ax.YLim = [lower_x upper_x], ax.XLim = [lower_y upper_y];
    subplot 232, imagesc(im_true),colorbar(),title("Ground Truth"), axis image, colormap gray, ax = gca, ax.YLim = [lower_x upper_x], ax.XLim = [lower_y upper_y];
    subplot 233, imagesc(im_true_S),colorbar(),title("Inpainted Ground Truth"), axis image, colormap gray, ax = gca, ax.YLim = [lower_x upper_x], ax.XLim = [lower_y upper_y];
    subplot 234, imagesc(xmap.*mask),colorbar(),title("Masked Map estimate"), axis image, colormap gray, ax = gca, ax.YLim = [lower_x upper_x], ax.XLim = [lower_y upper_y];
    subplot 235, imagesc(xmap),colorbar(),title("Map estimate"), axis image, colormap gray, ax = gca, ax.YLim = [lower_x upper_x], ax.XLim = [lower_y upper_y];
    subplot 236, imagesc(xmap_S),colorbar(),title("Inpainted Map estimate"), axis image, colormap gray, ax = gca, ax.YLim = [lower_x upper_x], ax.XLim = [lower_y upper_y];
    sgtitle(strjoin(["Inpainting results with kernel",param_struct.Size_Gauss_kern]))
    forward_param_names       = strjoin([param_data.sig_noise,"noise",geom.ndetectors,"ndtct",geom.n_angles,"agls",geom.spacing,"grdsz"],"_");
    inpainting_params = strjoin([param_struct.Size_Gauss_kern,"kernel"],"_");

    fig_name = strjoin(["inpainting_results",name,inpainting_params,forward_param_names,"fig"],["_","_","_","."]);
    png_fig_name = strjoin(["inpainting_results",name,inpainting_params,forward_param_names,"png"],["_","_","_","."]);
    fig_path = strjoin([target_folder,fig_name],"/");
    png_fig_path = strjoin([target_folder,'png',png_fig_name],"/");
    saveas(fig,fig_path)
    saveas(fig,png_fig_path)








end





nhood = [ 0 1 0; 1 1 1; 0 1 0];
se = strel(nhood)
mask_eroded = imerode(mask,se);
figure();imshow(mask_eroded)

subplot 121, imagesc(mask),colorbar(),title("Mask"), axis image, colormap gray, ax = gca, ax.YLim = [lower_x upper_x], ax.XLim = [lower_y upper_y];
subplot 122, imagesc(mask_eroded),colorbar(),title("Mask eroded"), axis image, colormap gray, ax = gca, ax.YLim = [lower_x upper_x], ax.XLim = [lower_y upper_y];





for i=1:size(kernels,2)
    param_struct.Size_Gauss_kern = cell2mat(kernels(i)) ;
    disp('create inpainting operator...')
    disp('     -- recursive - can take few minutes... --')
    param_struct.L = create_inpainting_operator_test(mask_eroded, param_struct.Size_Gauss_kern, xmap) ;
    disp('...done')
    param_struct.L = sparse(param_struct.L) ;




    %%
    



    %%




    xmap_S = xmap ;
    xmap_S(mask_eroded>0) = param_struct.L*xmap(mask_eroded==0);


    im_true_S = im_true;
    im_true_S(mask_eroded>0) = param_struct.L*im_true_S(mask_eroded==0);

    fig = figure(100), 
    pause(0.1);
    frame_h = get(handle(gcf),'JavaFrame');
    set(frame_h,'Maximized',1)
    subplot 231, imagesc(im_true.*mask_eroded),colorbar(),title("Eroded Masked Ground Truth"), axis image, colormap gray, ax = gca, ax.YLim = [lower_x upper_x], ax.XLim = [lower_y upper_y];
    subplot 232, imagesc(im_true),colorbar(),title("Ground Truth"), axis image, colormap gray, ax = gca, ax.YLim = [lower_x upper_x], ax.XLim = [lower_y upper_y];
    subplot 233, imagesc(im_true_S),colorbar(),title("Inpainted Ground Truth"), axis image, colormap gray, ax = gca, ax.YLim = [lower_x upper_x], ax.XLim = [lower_y upper_y];
    subplot 234, imagesc(xmap.*mask_eroded),colorbar(),title("Eroded Masked Map estimate"), axis image, colormap gray, ax = gca, ax.YLim = [lower_x upper_x], ax.XLim = [lower_y upper_y];
    subplot 235, imagesc(xmap),colorbar(),title("Map estimate"), axis image, colormap gray, ax = gca, ax.YLim = [lower_x upper_x], ax.XLim = [lower_y upper_y];
    subplot 236, imagesc(xmap_S),colorbar(),title("Inpainted Map estimate"), axis image, colormap gray, ax = gca, ax.YLim = [lower_x upper_x], ax.XLim = [lower_y upper_y];
    sgtitle(strjoin(["Inpainting results with kernel",param_struct.Size_Gauss_kern]))
    forward_param_names       = strjoin([param_data.sig_noise,"noise",geom.ndetectors,"ndtct",geom.n_angles,"agls",geom.spacing,"grdsz"],"_");
    inpainting_params = strjoin([param_struct.Size_Gauss_kern,"kernel"],"_");

    fig_name = strjoin(["eroded_inpainting_results",name,inpainting_params,forward_param_names,"fig"],["_","_","_","."]);
    png_fig_name = strjoin(["eroded_inpainting_results",name,inpainting_params,forward_param_names,"png"],["_","_","_","."]);
    fig_path = strjoin([target_folder,fig_name],"/");
    png_fig_path = strjoin([target_folder,'png',png_fig_name],"/");
    saveas(fig,fig_path)
    saveas(fig,png_fig_path)








end









