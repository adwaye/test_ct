


folder_name = "/home/amr62/matlab_projects/test_CT/Figures/check_xmaps";

ct_loc = "./Data/ct_scans/ct1";
slice_name = "curated2_pe_yslice_266";%"curated2_pe_yslice_266.mat", "curated2_pe_xslice_225
ct_mat = load(strjoin([ct_loc,slice_name,'mat'],["/","."]));


im_true = double(ct_mat.CT);


normA   = im_true-min(im_true(:));
normA   = normA./max(normA(:));
im_true = normA;


im_true = imresize(im_true,0.5,'Method','bilinear');


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



detector_setup = [450,450,450,450];
angle_setup    = [450,300,200,100];
noise_array    = [0.00005,0.0002,0.0003,0.001];
alpha_array  = [0.01];

noise      = noise_array(3);
ndetectors = detector_setup(2);
n_angles   = angle_setup(3);
spacing    = view_size/ndetectors;



forward_param_names = strjoin([noise,"noise",ndetectors,"ndtct",n_angles,"agls",spacing,"grdsz"],"_");
results_name        = strjoin([slice_name,"forward_problem_results",forward_param_names,"mat"],["_","_","."]);
file_name           = strjoin([folder_name,results_name],'/');

matfile = load(file_name);







figure(1), 
subplot(121),imagesc(im_true), axis image, colormap gray, colorbar, title("Ground Truth")
subplot(122),imagesc(matfile.xmap), axis image, colormap gray, colorbar, title("Simluated reconstruction")




