% Test UQ for MRI
% discrete Fourier measurements + wavelet
%% ------------------------------------------------------------------------

% clear all
% clc
% close all


addpath Algos/
addpath Data/
addpath Tools/
addpath(genpath('Tools'))

grad_norm_name = "L2"; %choose from L1, L2, Linf
M_norm_name    = "L2"; %choose from L1, L2, Linf
inpainting = false;%
use_dil_mask= false;
plot_all = false;

grad_scheme_name = "";%bw_fw or ""
algo_name = strjoin(["POCS",grad_norm_name,"gradM",M_norm_name,"Mx"],'_');
BUQO_algo = str2func(algo_name);
use_full_size = false;


source_folder = "~/matlab_projects/test_CT/Data/artefacts";
if grad_scheme_name ~= ""
    grad_op_name = strjoin(["gradient_op",grad_scheme_name],"_");
    folder_name   = strjoin(["artefact_experiment",grad_norm_name,"gradM",grad_scheme_name,M_norm_name,"M"],'_');
else
    grad_op_name = "gradient_op";
    folder_name   = strjoin(["artefact_experiment",grad_norm_name,"gradM",M_norm_name,"M"],'_');
end

if use_full_size
    folder_name = strjoin(["full_size",folder_name],'_');
end
if use_dil_mask
    folder_name = strjoin([folder_name,"dil_mask"],'_');
end
grad_op   = str2func(grad_op_name);
BUQO_algo = str2func(algo_name);



% folder_name = "~/matlab_projects/test_CT/Figures/ct1_experiment_L2_gradM_L2_M";
% folder_name = "~/matlab_projects/test_CT/Data/artefacts";
ct_loc = "./Data/ct_scans/ct1";
slice_name = "curated2_pe_yslice_266";%"curated2_pe_yslice_266.mat", "curated2_pe_xslice_225
ct_mat = load(strjoin([ct_loc,slice_name,'mat'],["/","."]));





% detector_setup = [450,450,450,450];
% angle_setup    = [450,300,200,100];
% noise_array    = [0.1];
% alpha_array  = [0.01];
% 
% noise      = 0.175;%noise_array(1);
% ndetectors = 450;%detector_setup(4);
% n_angles   = 50;%angle_setup(1);
% spacing    = view_size/ndetectors;
% 
% 
% 
% forward_param_names = strjoin([noise,"noise",ndetectors,"ndtct",n_angles,"agls",spacing,"grdsz"],"_");
% results_name        = strjoin([slice_name,"forward_problem_results",forward_param_names,"mat"],["_","_","."]);
% file_name           = strjoin([source_folder,results_name],'/');

%% 


target_folder = strjoin(["/home/adwaye/matlab_projects/test_CT/Figures",folder_name],'/');
mkdir(target_folder);
mkdir(strjoin([target_folder,"png"],'/'))

query     = strjoin([source_folder,"*slice_*"],'/');
filenames = dir(query);
nfiles    = size(filenames,1);
max_size = 350;

seed = 0 ;

for f = 1:nfiles
    tempname = filenames(f).name;
    name = tempname(1:22)
    file_name = strjoin([source_folder,tempname],'/');
    matfile = load(file_name);
    detector_setup = find_params(tempname);
    im_true = matfile.im_true;
    

%%
    param_data.Ny = max_size;
    param_data.Nx  = max_size;
    param_data.N = max_size*max_size ;

    
    

    geom.n_angles   = detector_setup('agls');%900;
    geom.ndetectors = 450;%900;
    

    vol_geom = astra_create_vol_geom(param_data.Nx,param_data.Ny);
    %create a parallel projection geometry with spacing =1 pixel, 384 


    geom.spacing    = detector_setup('grdsz');%1/sqrt(2);% 1/(nx*ny);


    geom.angles     = linspace2(0,pi,geom.n_angles);%use more angles
    proj_geom       = astra_create_proj_geom('parallel', geom.spacing, geom.ndetectors, geom.angles);
    geom.proj       = proj_geom;
    geom.vol        = vol_geom;
    stddev          = 1e-4;
    W = opTomo('cuda',proj_geom,vol_geom);
    W_scaled = W/W.normest;

    %num_angles = floor(1e-3*param_data.N) ;




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

    % mask_copy = mask;
    if plot_all
        figure(1), 
        subplot 131, imagesc(im_true), axis image, colormap gray, colorbar, title("Ground Truth")%,ax = gca, ax.YLim = [lower_x upper_x], ax.XLim = [lower_y upper_y];
        subplot 132, imagesc(mask_struct), axis image, colormap gray, colorbar, title("pe locations")%,ax = gca, ax.YLim = [lower_x upper_x], ax.XLim = [lower_y upper_y];
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
    param_data.sig_noise = detector_setup('noise');
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
%     if isfile(results_path)
%         load(results_path)
%     else
%         [xmap, fid, reg, norm_it, snr_it, time_it,time_total] = MAP_primal_dual(param_data, param_map, tau, sig1, sig2, max_it, stop_it, stop_norm, SNR) ;    
%         save(results_path,'xmap','fid','reg','norm_it','snr_it','time_it','time_total')
%     end
    xmap = double(matfile.xmap);
    % 





    if plot_all
        fig = figure, 
        subplot 221, imagesc(im_true), axis image, colormap gray, colorbar, xlabel('true')
        subplot 222, imagesc(mask_struct), axis image, colormap gray, colorbar, xlabel('mask')
        subplot 223, imagesc(im_fbp), axis image, colormap gray, colorbar, xlabel(['FBP - SNR ', num2str(SNR(im_fbp))])
        subplot 224, imagesc(xmap), axis image, colormap gray, colorbar, xlabel(['xmap - SNR ', num2str(snr_it(end))])
    end

%     fig = figure,
%     subplot 131, imagesc(im_true), axis image, colormap gray, colorbar, xlabel(['FBP - SNR ', num2str(SNR(im_fbp))])
%     subplot 132, imagesc(xmap), axis image, colormap gray, colorbar, xlabel(['xmap - SNR ', num2str(snr_it(end))])
%     subplot 133, imagesc(xmap.*(1-mask)), axis image, colormap gray, colorbar, xlabel(['xmap - SNR ', num2str(snr_it(end))])

    %%

    param_hpd.lambda_t = param_data.N / sum(abs(param_map.Psit(xmap))) ;


    alpha = 0.01;
    talpha = sqrt( (16*log(3/alpha)) / param_data.N );
    HPDconstraint = param_hpd.lambda_t* sum(abs(param_map.Psit(xmap))) ...
                    + param_data.N*(1+talpha);
    param_hpd.HPDconstraint = HPDconstraint/param_hpd.lambda_t ;


    param_hpd.Psit = param_map.Psit ;
    param_hpd.Psi = param_map.Psi ;
    param_hpd.normPsi = param_map.normPsi ;
    param_hpd.lambda = param_map.lambda ;



    mask_struct = matfile.mask_struct;
    [row_mask col_mask] = find(mask_struct);
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
    subplot 221, imagesc(mask_struct),colorbar(),title("mask"), axis image, colormap gray
    subplot 222, imagesc(tmp),colorbar(),title("reverse masked image"), axis image, colormap gray
    subplot 223, imagesc(im_true),colorbar(),title("ground truth"), axis image, colormap gray
    subplot 224, imagesc(xmap),colorbar(),title("map estimate"), axis image, colormap gray
    fig_name = strjoin([name,"fwd_res",forward_param_names,"fig"],["_","_","."]);
    fig_path = strjoin([target_folder,fig_name],"/");
    saveas(fig,fig_path)
    close(fig)


    %imwrite(im_true,'figures/g_truth')



    %===========================================================================
    % RUNNING BUQO NOW
    %===========================================================================

    mean_values_grad  = [0,0,0,0,0];
    quantiles    = [0.6,0.7,0.9];
    param_struct.Mask       = mask_struct; % Mask to select structure to test
    param_struct.choice_set = 'l2_const' ;
    
    
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

    disp('-------------------------------------------')
    [Gradop, Gradopt] = grad_op(rand(size(im_true)),mask_struct) ;
    disp("Testing if the gradient forward op and the divergence are true adjoints")
    disp(["Test object has size =",num2str(size(im_true))])
    disp(["artefact has area =",num2str(sum(mask_struct(:)))])
    xtmp  = rand(param_data.Ny,param_data.Nx) ;
    ytmp  = Gradop(rand(param_data.Ny,param_data.Nx)) ;% ytmp  = param_data.Phi(rand(param_data.Ny,param_data.Nx)) ;
    disp(["masked gradient op range dimension =",num2str(size(ytmp))])
    Pxtmp = Gradop(xtmp) ;% Pxtmp = param_data.Phi(xtmp) ;
    Ptytmp = Gradopt(ytmp) ;% Ptytmp = param_data.Phit(ytmp) ;
    fwd = Pxtmp(:)'*ytmp(:) ;
    bwd = xtmp(:)'* Ptytmp(:) ;
    disp('test adjoint operator')
    disp(['fwd gradmasked= ', num2str(fwd)])
    disp(['bwd divmasked= ', num2str(bwd)])
    disp(['diff = ', num2str(norm(fwd-bwd)/norm(fwd))])
    disp('-------------------------------------------')


    param_struct.Gradop = Gradop ;
    param_struct.Gradopt = Gradopt;
%     param_struct.Mbarop = Mask_op - param_struct.L*Mask_op_comp;
    param_struct.Mask_op = Mask_op;

    param_struct.l2b_Mask = sqrt(sum(param_struct.Mask(:)) + 2* sqrt(sum(param_struct.Mask(:)))) *  param_data.sig_noise ;
    xM_ = Mask_op * xmap(:) ;
    param_struct.tol_smooth = std(xM_) ; %1e-5  ;AR note: this is [-tau,+tau]


    param_struct.Psi = param_hpd.Psi ;
    param_struct.Psit = param_hpd.Psit ;
    param_struct.normPsi = param_hpd.normPsi ;
    param_struct.l1bound = param_hpd.HPDconstraint ; 


    param_algo.NbIt = 10000 ;
    param_algo.stop_dist = 1e-4 ;
    param_algo.stop_norm2 = 1e-4 ;
    param_algo.stop_norm1 = 1e-4 ;
    param_algo.stop_err_smooth = 1e-4 ;


    max_l = size(quantiles,2);
    texture_mask                  = matfile.texture_mask;
    

    [fx, fy] = gradient(xmap);
    sampled_gradients = [fx(texture_mask>0),fy(texture_mask>0)];


    sampled_pixels = xmap(texture_mask>0);

    mean_values_M  = [median(sampled_pixels(:)),median(sampled_pixels(:)),median(sampled_pixels(:)),median(sampled_pixels(:)),median(sampled_pixels(:))];



    max_l = size(quantiles,2);




    for bound_num=1:max_l
        param_struct.l2_mean_pix = median(sampled_pixels(:));%center of the ball for M
        pix_quantile = quantiles(bound_num);
        param_struct.l2_bound_pix = max([quantile(sampled_pixels(:),pix_quantile)-median(sampled_pixels(:)),median(sampled_pixels(:))-quantile(sampled_pixels(:),1-pix_quantile)]);%radius of the ball for M

        param_struct.l2_mean_grad = 0.0;%center of the ball for the gradient
        grad_quantile = quantiles(bound_num);
        param_struct.l2_bound_grad= max([quantile(sampled_gradients(:),grad_quantile)-median(sampled_gradients(:)),median(sampled_gradients(:))-quantile(sampled_gradients(:),1-grad_quantile)]);%radius of the ball for the gradient



        disp("**************************************************************")
        disp("Finding projection of xmap in S xmap_S")
        proj_S = Project_to_S(xmap, param_algo, param_data,  param_struct);
        xmap_S = proj_S.xS ;
        l1_mapS = sum(abs(param_map.Psit(xmap_S))) ;
        l2_mapS = sqrt(sum( abs( param_data.Phi(xmap_S) - param_data.y ).^2 ,'all')) ;

        %             xmap_S(param_struct.Mask>0) = param_struct.L*xmap(param_struct.Mask==0) ;
        if plot_all
            fig=figure, 
            subplot 221, imagesc(tmp), axis image; colorbar, colormap gray, xlabel('map without struct')
            subplot 222, imagesc(xmap_S), axis image ; colorbar, colormap gray, xlabel('smoothed struct')
            subplot 223, imagesc(im_true), axis image; colorbar, colormap gray, xlabel('true')
            subplot 224, imagesc(xmap), axis image; colorbar, colormap gray, xlabel('map')
        end
        %% finding xC and xS
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
            result = BUQO_algo(xmap, xmap_S, param_algo, param_data, param_hpd, param_struct) ;
            sigma4 = 1;
        end





        if plot_all
            fig=figure, 
            subplot 221, imagesc(tmp), axis image; colorbar, colormap gray, xlabel('map without struct')
            subplot 222, imagesc(xmap_S), axis image ; colorbar, colormap gray, xlabel('smoothed struct')
            subplot 223, imagesc(im_true), axis image; colorbar, colormap gray, xlabel('true')
            subplot 224, imagesc(xmap), axis image; colorbar, colormap gray, xlabel('map')
        end
        %param_struct.l2_mean = 0 ;





        rho  = norm( result.xC(:) - result.xS(:) ) / ...
               norm( xmap(:) - xmap_S(:) ) ;
        disp(' ')
        disp('*****************************************')
        disp(['       rho = ',num2str(rho)])
        disp('*****************************************')


        l2_bound_params       = strjoin([param_struct.l2_mean_grad,grad_norm_name,"mean_grad",param_struct.l2_bound_grad,grad_norm_name,"bound_grad",grad_quantile,"grad_quantile",param_struct.l2_mean_pix,M_norm_name,"mean_pix",param_struct.l2_bound_pix,M_norm_name,"bound_pix",pix_quantile,"pix_quantile",alpha,"alpha"],"_");
    %     l2_bound_params_fig   = strjoin([param_struct.l2_mean_grad,grad_norm_name,"mean_grad",param_struct.l2_bound_grad,grad_norm_name,"bound_grad",param_struct.l2_mean_pix,M_norm_name,"mean_pix",param_struct.l2_bound_pix,M_norm_name,"bound_pix",alpha,"alpha"]," ");
        l2_bound_params_fig   = strjoin([grad_norm_name,"ball on GradM center=",param_struct.l2_mean_grad,"quantile=",grad_quantile,"radius=",param_struct.l2_bound_grad,"|",...
                                         M_norm_name,"ball on M center=",param_struct.l2_mean_pix,"radius=",param_struct.l2_bound_pix,"quantile=",grad_quantile,"|"," alpha=",alpha]," ");
    %     linf_bound_params = strjoin([param_struct.linf_bound_max,"linfMax",param_struct.linf_bound_min,"linfMin",sigma4,"sigma4"],"_");
    %     linf_bound_params_fig = strjoin([param_struct.linf_bound_max,"linfMax",param_struct.linf_bound_min,"linfMin",sigma4,"sigma4"],"-");

        if grad_norm_name == "Linf"
            fig_name     = strjoin([name,"BUQO_problem_results",linf_bound_params,forward_param_names,"fig"],["_","_","_","."]);
            png_fig_name = strjoin([name,"BUQO_problem_results",linf_bound_params,forward_param_names,"png"],["_","_","_","."]);
    %         sup_title    = strjoin([inpainting_ext,linf_bound_params_fig],"-");
            sup_title    = linf_bound_params_fig;
        else
            fig_name     = strjoin([name,"BUQO_problem_results",l2_bound_params,forward_param_names,"fig"],["_","_","_","."]);
            png_fig_name = strjoin([name,"BUQO_problem_results",l2_bound_params,forward_param_names,"png"],["_","_","_","."]);
    %         sup_title    = strjoin([inpainting_ext,l2_bound_params_fig],"-");
            sup_title    = l2_bound_params_fig;
        end

        fig_path     = strjoin([target_folder,fig_name],"/");
        png_fig_path = strjoin([target_folder,'png',png_fig_name],"/");


        fig =figure;
        pause(0.00001);
        frame_h = get(handle(gcf),'JavaFrame');
        set(frame_h,'Maximized',1);
        subplot 221, imagesc(xmap), axis image; colormap gray; colorbar, xlabel('xmap'), ax = gca, ax.YLim = [lower_x upper_x], ax.XLim = [lower_y upper_y];
        subplot 222, imagesc(xmap_S), axis image; colormap gray; colorbar, xlabel('xmap - no struct'), ax = gca, ax.YLim = [lower_x upper_x], ax.XLim = [lower_y upper_y];
        subplot 223, imagesc(result.xC), axis image; colormap gray; colorbar, xlabel('xC'), ax = gca, ax.YLim = [lower_x upper_x], ax.XLim = [lower_y upper_y];
        subplot 224, imagesc(result.xS), axis image; colormap gray; colorbar, xlabel('xS'), ax = gca, ax.YLim = [lower_x upper_x], ax.XLim = [lower_y upper_y];

    %     sgtitle(sup_title)

        sgtitle(strjoin([sup_title,' rho=',rho,' angles= ',geom.n_angles,' detectors= ',geom.ndetectors],''))
        disp(['saving everything as',strjoin([name,"BUQO_problem_results",l2_bound_params,forward_param_names],'_')]);

        saveas(fig,fig_path)


        saveas(fig,png_fig_path)
        Mask = param_struct.Mask;


        lambda_t = param_hpd.lambda_t;

        hpd_constraint = param_hpd.HPDconstraint;
        epsilon        = param_data.data_eps;
    %     theta          = param_struct.l2_bound;
        l2_bound_pix   = param_struct.l2_bound_pix;
        l2_mean_pix    = param_struct.l2_mean_pix;
        l2_mean_grad   = param_struct.l2_mean_grad;
        l2_bound_grad  = param_struct.l2_bound_grad;
    %     tau            = param_struct.tol_smooth;

        phi_imtrue     = param_data.Phi(im_true);
        phit_imtrue    = param_data.Phit(phi_imtrue);
        psi_imtrue     = param_map.Psit(im_true);
        psit_imtrue     = param_map.Psi(psi_imtrue);
        x_c            = result.xC;
        x_s            = result.xS;
        struct_mask   = param_struct.Mask;

        if isfield(result,'time')
            time_tot = result.time;
        else
            time_tot = [0];
        end
        if isfield(result,'dist2')
            dist2 = result.dist2  ;
        else
            dist2 = [0];
        end
        if isfield(result,'l2data')
            l2data = result.l2data ;
        else
            l2data = [0];
        end
        if isfield(result,'l1reg')
            l1reg = result.l1reg   ;
        else
            l1reg = [0];
        end

        if grad_norm_name == "Linf"
            if isfield(result,'linfsmooth_min')
                linfsmooth_min = result.linfsmooth_min ; %traceplot of how the max of gradxs is evolving
            else
                linfsmooth_min =[0];
            end
            if isfield(result,'linfsmooth_max')
                linfsmooth_max = result.linfsmooth_max;  %traceplot of how the min of gradxs is evolving
            else
                linfsmooth_max = [0];
            end
            fig_name     = strjoin([name,"BUQO_problem_convergence",linf_bound_params,forward_param_names,"fig"],["_","_","_","."]);
            results_name = strjoin([name,"BUQO_problem_results",linf_bound_params,forward_param_names,"mat"],["_","_","_","."]);
        else
            if isfield(result,'l2smooth')
                l2smooth     = result.l2smooth ;%traceplot of how the norm of gradxs is evolving
            else
                l2smooth = [0];
            end
            if isfield(result,'smooth_max')
                smooth_max = result.smooth_max ;
            else
                smooth_max =[0];
            end
            fig_name     = strjoin([name,"BUQO_problem_convergence",l2_bound_params,forward_param_names,"fig"],["_","_","_","."]);
            results_name = strjoin([name,"BUQO_problem_results",l2_bound_params,forward_param_names,"mat"],["_","_","_","."]);
        end

        results_path        = strjoin([target_folder ,results_name],'/');




        save(results_path,'xmap','hpd_constraint','l2_bound_pix','l2_bound_grad','l2_mean_pix','l2_mean_grad','tau','epsilon','struct_mask','phi_imtrue','x_c','x_s','dist2','l2data','l1reg','l2smooth','rho','smooth_max','alpha')

        fig =figure, 
        subplot(321);plot(l1reg), hold on, plot(hpd_constraint*ones(size(l1reg)),'r'),xlabel('it'), ylabel('l1 norm regulariser hpd-psi xc'), 
        subplot(322);plot(l2data), hold on, plot(epsilon*ones(size(l2data)),'r'), xlabel('it'), ylabel('l2 norm data fit-phi xc')
        subplot(323);plot(l2smooth), hold on, plot(l2_bound_pix*ones(size(l2smooth)),'r'),  xlabel('it'), ylabel(strjoin([M_norm_name,' Mxs']))
        subplot(324);plot(smooth_max), hold on, plot(l2_bound_grad*ones(size(smooth_max)),'r'),  xlabel('it'), ylabel(strjoin([grad_norm_name,' grad Mxs']))
        subplot(325);plot(dist2),   xlabel('it'), ylabel('Distance between xc and xs')






        if grad_norm_name == "Linf"
            fig_name     = strjoin([name,"BUQO_problem_convergence",linf_bound_params,forward_param_names,"fig"],["_","_","_","."]);
        else
            fig_name     = strjoin([name,"BUQO_problem_convergence",l2_bound_params,forward_param_names,"fig"],["_","_","_","."]);
        end
        fig_path = strjoin([target_folder,fig_name],"/");
        sgtitle(strjoin([sup_title,' rho=',rho,' angles=',geom.n_angles,' detectors=',geom.ndetectors],''))
        saveas(fig,fig_path)

        
    end
end
close all

    % mask_dil = imdilate(mask,strel('disk',2));
    % figure();imagesc(mask_dil-mask)
