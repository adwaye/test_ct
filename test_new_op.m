

clear all
clc
close all


addpath Algos/
addpath Data/
addpath Tools/
addpath(genpath('Tools'))

norm_name = "L2"; %choose from L1, L2, Linf
inpainting = false;%
if inpainting
    inpainting_ext = "Lx";
else
    inpainting_ext = "noLx";
end

algo_name = strjoin(["POCS",norm_name,"gradM",inpainting_ext],'_');
BUQO_algo = str2func(algo_name);



%% 
source_folder = "Data/ct_scans/ct1";
folder_name   = strjoin(["ct1_experiment",norm_name,"gradM",inpainting_ext],'_');
target_folder = strjoin(["/home/adwaye/matlab_projects/test_CT/Figures",folder_name],'/');
mkdir(target_folder);
mkdir(strjoin([target_folder,"png"],'/'))

query     = strjoin([source_folder,"*slice_*"],'/');
filenames = dir(query);
nfiles    = size(filenames,1);

% tempname = "curated2_pe_zslice_189.mat";%$change this to process a different slice
tempname = "curated2_pe_xslice_225.mat";
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

[param_data.Ny,param_data.Nx] = size(im_true) ;
param_data.N = numel(im_true) ;
param_data.Nx = size(im_true,1);
param_data.Ny = size(im_true,2);
disp('-------------------------------------------')
[Mop, Mopt] = gradient_op_bw_fw(rand(size(im_true)),mask) ;
disp("Testing if the gradient forward op and the divergence are true adjoints")
disp(["Test object has size =",num2str(size(im_true))])
disp(["artefact has area =",num2str(sum(mask(:)))])
xtmp  = rand(param_data.Ny,param_data.Nx) ;
ytmp  = Mop(rand(param_data.Ny,param_data.Nx)) ;% ytmp  = param_data.Phi(rand(param_data.Ny,param_data.Nx)) ;
disp(["masked gradient op range dimension =",num2str(size(ytmp))])
Pxtmp = Mop(xtmp) ;% Pxtmp = param_data.Phi(xtmp) ;
Ptytmp = Mopt(ytmp) ;% Ptytmp = param_data.Phit(ytmp) ;
fwd = Pxtmp(:)'*ytmp(:) ;
bwd = xtmp(:)'* Ptytmp(:) ;
disp('test adjoint operator')
disp(['fwd gradmasked= ', num2str(fwd)])
disp(['bwd divmasked= ', num2str(bwd)])
disp(['diff = ', num2str(norm(fwd-bwd)/norm(fwd))])
disp('-------------------------------------------')