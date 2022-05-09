

view_size = 1.05*sqrt(size(256,1)^2+size(256,2)^2);
ndetectors = 4000;
n_angles=10000;
spacing = view_size/geom.ndetectors;
spacing = 0.079385;
sig_noise = 0.0001;
forward_param_names       = strjoin([sig_noise,"noise",ndetectors,"ndtct",n_angles,"agls",spacing,"grdsz"],"_");
results_name        = strjoin([name,"forward_problem_results",forward_param_names,"mat"],["_","_","."]);

file_name = strjoin(["/home/adwaye/matlab_projects/test_CT/Figures/ct1_experiment_L2_gradM_L2_M_dil_mask/",results_name],'')
matfile   = load(file_name);

M = 2000*1000;
sig_noise = 0.0005;
eps  = sqrt(2*M + 2* sqrt(4*M)) *  sig_noise ;
fig =figure, 
subplot(131);plot(matfile.fid), hold on, plot(eps*ones(size(matfile.fid)),'r'), xlabel('it'), ylabel('Data fit term'), 
subplot(132);plot(matfile.reg),  xlabel('it'), ylabel('l1 norm data regulariser')
subplot(133);imagesc(matfile.xmap) ,axis image, colormap gray, colorbar, title("xmap")

