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
filename = "Data/ct_scans/ct1/pe_zslice_213.mat";
target_folder = "/home/adwaye/matlab_projects/test_CT/Figures/ct1";
matfile = load(filename);
im_true = double(matfile.CT);
normA   = im_true-min(im_true(:));
normA   = normA./max(normA(:));
im_true = normA;
im_true = imresize(im_true,[256,256],'Method','bilinear');
im_true = padarray(im_true,[128 128],0,'both');

mask = matfile.labels;
mask = imresize(mask,0.5,'bilinear');
mask = double(mask>0);
mask = padarray(mask,[128 128],0,'both');

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

sig1 = 0.7/param_map.normPsi ;
sig2 = 0.7/param_data.normPhi ;
tau = 0.99 / (sig1*param_map.normPsi + sig2*param_data.normPhi) ;
disp(['tau = ', num2str(tau)])
disp(['sig1 = ', num2str(sig1)])
disp(['sig2 = ', num2str(sig2)])

max_it = 20000 ;
stop_it = 1e-4 ;
stop_norm = 1e-4 ;



[filepath,name,ext] = fileparts(filename);

results_name        = strjoin([name,"forward_problem_results.mat"],"_");
results_path        = strjoin([target_folder ,results_name],'/');
if isfile(results_path)
    load(results_path)
    
else
    [xmap, fid, reg, norm_it, snr_it, time_it,time_total] = MAP_primal_dual(param_data, param_map, tau, sig1, sig2, max_it, stop_it, stop_norm, SNR) ;    
    save(results_path,'xmap','fid','reg','norm_it','snr_it','time_it','time_total')
end
xmap = double(xmap);
% 






fig = figure, 
subplot 221, imagesc(im_true), axis image, colormap gray, colorbar, xlabel('true')
subplot 222, imagesc(mask), axis image, colormap gray, colorbar, xlabel('mask')
subplot 223, imagesc(im_fbp), axis image, colormap gray, colorbar, xlabel(['FBP - SNR ', num2str(SNR(im_fbp))])
subplot 224, imagesc(xmap), axis image, colormap gray, colorbar, xlabel(['xmap - SNR ', num2str(snr_it(end))])
fig_ext = strjoin([name,"fwd_res",prod(size(norm_it)),"it",geom.ndetectors,"ndtct",geom.n_angles,"agls",geom.spacing,"grdsz.fig"],"_")
fig_path = strjoin([target_folder,fig_ext],"/")
saveas(fig,fig_path)



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


mask_struct= single(mask>0);
tmp = xmap ; tmp(mask_struct>0) = 0 ;

figure(99), 
subplot 221, imagesc(mask),colorbar(),title("mask"), axis image, colormap gray
subplot 222, imagesc(tmp),colorbar(),title("reverse masked image"), axis image, colormap gray
subplot 223, imagesc(im_true),colorbar(),title("ground truth"), axis image, colormap gray
subplot 224, imagesc(xmap),colorbar(),title("map estimate"), axis image, colormap gray


%imwrite(im_true,'figures/g_truth')



%===========================================================================
% RUNNING BUQO NOW
%===========================================================================


param_struct.Mask       = mask_struct; % Mask to select structure to test
param_struct.choice_set = 'l2_const' ;
param_struct.Size_Gauss_kern = [3,7,11] ;
disp('create inpainting operator...')
disp('     -- recursive - can take few minutes... --')
param_struct.L = create_inpainting_operator_test(param_struct.Mask, param_struct.Size_Gauss_kern, xmap) ;
disp('...done')
param_struct.L = sparse(param_struct.L) ;

param_struct.Lbar = sparse([-speye(sum(param_struct.Mask(:))), param_struct.L]) ;
param_struct.Nout = sum(param_struct.Mask(:)==0) ;
param_struct.normL = op_norm(@(x) param_struct.L*x, @(x) param_struct.L'*x, [param_struct.Nout,1], 1e-4, 200, 0);  
param_struct.normLbar = op_norm(@(x) param_struct.Lbar*x, @(x) param_struct.Lbar'*x, [numel(param_struct.Mask),1], 1e-4, 200, 0);  

param_struct.Li =@(u) [param_struct.L*u ; u] ;
param_struct.Lit =@(x) param_struct.L'*x(param_struct.Mask>0) + x(param_struct.Mask==0) ;
param_struct.normLi = op_norm(param_struct.Lit, param_struct.Li, [numel(param_struct.Mask),1], 1e-4, 200, 0);  

xmap_S = xmap ;
xmap_S(param_struct.Mask>0) = param_struct.L*xmap(param_struct.Mask==0) ;
figure, 
subplot 221, imagesc(tmp), axis image; colorbar, colormap gray, xlabel('map without struct')
subplot 222, imagesc(xmap_S), axis image ; colorbar, colormap gray, xlabel('smoothed struct')
subplot 223, imagesc(im_true), axis image; colorbar, colormap gray, xlabel('true')
subplot 224, imagesc(xmap), axis image; colorbar, colormap gray, xlabel('map')

param_struct.l2_mean = 0 ;
param_struct.l2_bound = sqrt(sum(abs(xmap_S(param_struct.Mask>0)).^2)) ; %AR note: this is theta

Mop = sparse(sum(param_struct.Mask(:)), numel(param_struct.Mask)) ;
Mopcomp = sparse(numel(param_struct.Mask)-sum(param_struct.Mask(:)), numel(param_struct.Mask)) ;
i = 1; ic = 1;
for n = 1:numel(param_struct.Mask)
    if(param_struct.Mask(n))==0
        Mopcomp(ic,n) = 1 ;
        ic = ic+1 ;
    else
        Mop(i,n) = 1;
        i = i+1 ;
    end
end

param_struct.Mop = Mop ;
param_struct.Mbarop = Mop - param_struct.L*Mopcomp;

param_struct.l2b_Mask = sqrt(sum(param_struct.Mask(:)) + 2* sqrt(sum(param_struct.Mask(:)))) *  param_data.sig_noise ;
xM_ = param_struct.Mop * xmap(:) ;
param_struct.tol_smooth = std(xM_) ; %1e-5  ;AR note: this is [-tau,+tau]


param_struct.Psi = param_hpd.Psi ;
param_struct.Psit = param_hpd.Psit ;
param_struct.normPsi = param_hpd.normPsi ;
param_struct.l1bound = param_hpd.HPDconstraint ; 








%%

param_algo.NbIt = 10000 ;
param_algo.stop_dist = 1e-4 ;
param_algo.stop_norm2 = 1e-4 ;
param_algo.stop_norm1 = 1e-4 ;
param_algo.stop_err_smooth = 1e-4 ;



l1_mapS = sum(abs(param_map.Psit(xmap_S))) ;
l2_mapS = sqrt(sum( abs( param_data.Phi(xmap_S) - param_data.y ).^2 ,'all')) ;


disp(' ')
disp(' ')
disp(' ')
disp(' ')
disp(' ')
disp('*******************************************************')
disp('*******************************************************')
disp(['l1 inpaint = ',num2str(l1_mapS)])
disp(['HPD bound = ',num2str(param_hpd.HPDconstraint)])
disp(['l2 data inpaint = ',num2str(l2_mapS)])
disp(['data bound = ',num2str(param_data.data_eps)])

if l1_mapS <= param_hpd.HPDconstraint && l2_mapS <= param_data.data_eps
    disp('Intersection between S and Calpha nonempty')
    disp('*******************************************************')
    result.xS = xmap_S ;
    result.xC = xmap_S ;
else
    disp('xmap_S OUTSIDE Calpha -> run alternating projections')
    disp('*******************************************************')
    disp(' ')
    result = POCS_PD_global_relax(xmap, xmap_S, param_algo, param_data, param_hpd, param_struct) ;
end


rho  = norm( result.xC(:) - result.xS(:) ) / ...
       norm( xmap(:) - xmap_S(:) ) ;
disp(' ')
disp('*****************************************')
disp(['       rho = ',num2str(rho)])
disp('*****************************************')

fig =figure, 
subplot 221, imagesc(xmap), axis image; colormap gray; colorbar, xlabel('xmap')
subplot 222, imagesc(xmap_S), axis image; colormap gray; colorbar, xlabel('xmap - no struct')
subplot 223, imagesc(result.xC), axis image; colormap gray; colorbar, xlabel('xC')
subplot 224, imagesc(result.xS), axis image; colormap gray; colorbar, xlabel('xS')
fig_ext = strjoin([name,"BUQO_res",prod(size(result.time)),"it",geom.ndetectors,"ndtct",geom.n_angles,"agls",geom.spacing,"grdsz.fig"],"_")
fig_path = strjoin([target_folder,fig_ext],"/")
saveas(fig,fig_path)


L = param_struct.L;
Mask = param_struct.Mask;

%Mask = load("/home/adwaye/PycharmProjects/CT-UQ/adwaye_results/adwaye_mask.mat");
%Mask = Mask.mask;
% [L, xinp_tot,xinp_multscale]  = create_inpainting_operator_test(Mask, param_struct.Size_Gauss_kern, xmap);
% bar =@(x) x(Mask>0) - L*x(Mask==0) ;
% Msel = sparse(1:sum(Mask(:)), find(Mask(:)), ones(sum(Mask(:)),1), sum(Mask(:)), numel(Mask));
% Mcsel = sparse(1:sum(1-Mask(:)), find(1-Mask(:)), ones(sum(1-Mask(:)),1), sum(1-Mask(:)), numel(Mask));
% Lbar_ =@(x) (Msel - L * Mcsel) * x(:) ;
% Lbaradj_ =@(y) (Msel' - Mcsel' * L') * y ;


% normLbar = op_norm(Lbar_, Lbaradj_, size(xmap), 1e-3, 400, 0);


lambda_t = param_hpd.lambda_t;

hpd_constraint = param_hpd.HPDconstraint;
epsilon        = param_data.data_eps;
theta          = param_struct.l2_bound;
tau            = param_struct.tol_smooth;

phi_imtrue     = param_data.Phi(im_true);
phit_imtrue    = param_data.Phit(phi_imtrue);
psi_imtrue     = param_map.Psit(im_true);
psit_imtrue     = param_map.Psi(psi_imtrue);
x_c            = result.xC;
x_s            = result.xS;
Mop_imtrue     = param_struct.Mop*im_true(:);
Mopt =@(v) reshape(param_struct.Mop'*v, param_data.Ny, param_data.Nx);
Mopt_imtrue     = Mopt(Mop_imtrue(:));
Lbar_imtrue    = param_struct.Mbarop*im_true(:);
Mbaropt =@(v) reshape(param_struct.Mbarop'*v, param_data.Ny, param_data.Nx) ;
Lbart_imtrue = Mbaropt(Lbar_imtrue);
fft_imtrue   = TF(im_true,numel(im_true));
struct_mask   = param_struct.Mask;

time_tot = result.time;

dist2 = result.dist2  ;
l2data = result.l2data ;
l1reg = result.l1reg   ;
l2smooth = result.l2smooth ;
smooth_max = result.smooth_max ;




[filepath,name,ext] = fileparts(filename);

results_name        = strjoin([name,"Buqo_problem_results.mat"],"_");
results_path        = strjoin(["/home/adwaye/matlab_projects/test_CT/Figures/ct1" ,results_name],'/');
save(results_path,'xmap','hpd_constraint','theta','tau','epsilon','struct_mask','phi_imtrue','x_c','x_s','dist2','l2data','l1reg','l2smooth','smooth_max','rho')




