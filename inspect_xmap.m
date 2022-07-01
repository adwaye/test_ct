close all 
clear all 
clc

save_ = true;
%folder_name = "~/matlab_projects/test_CT/Figures/ct1_experiment_L2_gradM_L2_M";
folder_name = "~/matlab_projects/test_CT/Figures/ct1_experiment_L2_gradM_L2_M";
ct_loc = "./Data/ct_scans/ct1";
slice_name = "curated2_pe_yslice_266";%"curated2_pe_yslice_266.mat", "curated2_pe_xslice_225
ct_mat = load(strjoin([ct_loc,slice_name,'mat'],["/","."]));
CT = ct_mat.CT;

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
noise_array    = [0.1];
alpha_array  = [0.01];

noise      = 0.175;%noise_array(1);
ndetectors = 450;%detector_setup(4);
n_angles   = 50;%angle_setup(1);
spacing    = view_size/ndetectors;



forward_param_names = strjoin([noise,"noise",ndetectors,"ndtct",n_angles,"agls",spacing,"grdsz"],"_");
results_name        = strjoin([slice_name,"forward_problem_results",forward_param_names,"mat"],["_","_","."]);
file_name           = strjoin([folder_name,results_name],'/');

load(file_name);







figure(5), 
subplot(121),imagesc(im_true), axis image, colormap gray, colorbar, title("Ground Truth")
subplot(122),imagesc(xmap), axis image, colormap gray, colorbar, title("Simluated reconstruction")


if save_
%     target_folder = strjoin(["~/matlab_projects/test_CT/Figures/for_jonathan",results_name],'/');
%     mkdir(target_folder)
%     imwrite(matfile.xmap,strjoin([target_folder,"simulation.png"],"/"))
%     imwrite(im_true,strjoin([target_folder,"ground_truth.png"],"/"))
%     
    target_loc = "./Data/artefacts"
    disp(strjoin(["saving the artefacts to be used in ",target_loc]))
    mask_struct = zeros(size(xmap));
    texture_mask = zeros(size(xmap));
  
    
  if strmatch(results_name,"curated2_pe_xslice_225_forward_problem_results_0.001_noise_450_ndtct_100_agls_0.70564_grdsz.mat")
       
        mask_struct(150:160,185:189)=1;
        mask_struct(151:152,187:200)=1;
        mask_struct(157:158,185:198)=1;
        texture_mask(153:156,190:204)=1;
    
  elseif strmatch(results_name,"curated2_pe_xslice_225_forward_problem_results_0.175_noise_450_ndtct_50_agls_0.70564_grdsz.mat")
        mask_struct(153:162,183:189)=1;
        mask_struct(150:152,187:192)=1;
        mask_struct(151:152,196:200)=1;
        mask_struct(157:159,194:198)=1;
        
        
        texture_mask(150:163,185:200)=1;
        texture_mask(150:163,185:200)=1;
        texture_mask(155:160,180:185)=1;
        
        texture_mask(153:162,183:189)=0;
        texture_mask(150:152,187:192)=0;
        texture_mask(151:152,196:200)=0;
        texture_mask(157:159,194:198)=0;
  elseif strmatch(results_name,"curated2_pe_xslice_225_forward_problem_results_0.25_noise_450_ndtct_100_agls_0.70564_grdsz.mat")
      mask_struct(154:160,185:189)=1;
      mask_struct(150:152,186:199)=1;
        
        
      texture_mask(150:163,185:200)=1;
      texture_mask(150:163,185:200)=1;
      texture_mask(155:160,180:185)=1;
        
      texture_mask(154:160,185:189)=0;
      texture_mask(150:152,186:199)=0;
  elseif strmatch(results_name,"curated2_pe_yslice_266_forward_problem_results_0.175_noise_450_ndtct_50_agls_0.70564_grdsz.mat")

      mask_struct = zeros(size(xmap));
      texture_mask = zeros(size(xmap));
      mask_struct(181:186,203:204)  = 1;
      texture_mask(174:178,210:215) = 1;
      texture_mask(171:173,211:215) = 1;
      figure(5), 
      subplot(121),imagesc(xmap), axis image, colormap gray, colorbar, title("xmap ")
      subplot(122),imagesc((1-mask_struct).*xmap), axis image, colormap gray, colorbar, title("masked xmap-artefact")
        

      figure(6), 
      subplot(121),imagesc(xmap), axis image, colormap gray, colorbar, title("xmap ")
      subplot(122),imagesc((1-texture_mask).*xmap), axis image, colormap gray, colorbar, title("masked xmap-sampled pixels")
      splits = split(results_name,'_')
      prefix = strjoin(['curated3',splits(2),splits(3),splits(4)],'_')
      res_name = strjoin([prefix,"forward_problem_results",forward_param_names,"mat"],["_","_","."]);
      output_path = strjoin([target_loc,res_name],'/')
    
      save(output_path,'xmap','fid','reg','norm_it','snr_it','time_it','time_total',...
                     'texture_mask','mask_struct','im_true','CT')
      
        
      mask_struct = zeros(size(xmap));
      texture_mask = zeros(size(xmap));
      mask_struct(185:190,209:211)=1;
      texture_mask(183:190,210:215)=1;
      texture_mask(191:194,210:213)=1;
      texture_mask(185:190,209:211)=0;
        

    
    
  end
    
    figure(1), 
    subplot(121),imagesc(xmap), axis image, colormap gray, colorbar, title("xmap ")
    subplot(122),imagesc((1-mask_struct).*xmap), axis image, colormap gray, colorbar, title("masked xmap-artefact")
        

    figure(2), 
    subplot(121),imagesc(xmap), axis image, colormap gray, colorbar, title("xmap ")
    subplot(122),imagesc((1-texture_mask).*xmap), axis image, colormap gray, colorbar, title("masked xmap-sampled pixels")
    
    
    output_path = strjoin([target_loc,results_name],'/')
    save(output_path,'xmap','fid','reg','norm_it','snr_it','time_it','time_total',...
                     'texture_mask','mask_struct','im_true','CT')

end

