function results = POCS_PD_global_relax(xmap, xmap_inp, param_algo, param_data, param_hpd, param_struct)


%% Operateurs
if ~isfield(param_struct, 'back'), param_struct.back = zeros(size(xmap_inp)) ; end;
if ~isfield(param_algo, 'display'), param_algo.display = 100 ; end;

Mbarop =@(x) param_struct.Mbarop*x(:) ;
Mbaropt =@(v) reshape(param_struct.Mbarop'*v, param_data.Ny, param_data.Nx) ;

Mop =@(x) param_struct.Mop*x(:) ;
Mopt =@(v) reshape(param_struct.Mop'*v, param_data.Ny, param_data.Nx) ;

%% STEP-SIZES

sigma1 = 1/sqrt(param_data.normPhi+param_struct.normPsi) ;
sigma2 = sigma1 ;
normMbar = op_norm(Mbarop, Mbaropt, [param_data.Ny, param_data.Nx], 1e-4, 200, 0) ;
sigma3 = 1 / sqrt( normMbar ) ;
normM = op_norm(Mop, Mopt, [param_data.Ny, param_data.Nx], 1e-4, 200, 0) ;
sigma4 = 1 / sqrt( normM ) ;





tau = 0.9 / (1 + sigma1 * param_data.normPhi ...
               + sigma2 * param_struct.normPsi ...
               + sigma3 * normMbar ...
               + sigma4 * normM) ;

%save('/home/adwaye/PycharmProjects/CT-UQ/data/stepSizes.mat','sigma1','sigma2','sigma3','sigma4','tau');           
           
if tau * (1 + sigma1*param_struct.normPsi+ sigma2*param_data.normPhi+ sigma3*normMbar+ sigma4*normM) >=1
    disp('Error convergence parameters for HPD constraint')
end


%% INITIALISATION

xC = xmap ;
xS = xmap_inp ;

v1 = 0 * param_data.y ;
v2 = 0 * param_hpd.Psit(xC) ;

v3 = 0 * Mbarop(xS) ;
v4 = 0 * Mop(xS) ;


Phix = param_data.Phi(xC) ;
Psix = param_hpd.Psit(xC) ;
MbarxS = Mbarop(xS) ;
MxS = Mop(xS) ;


%% critere
dist2(1) = sum(abs(xS(:) - xC(:)).^2) ;
l2data(1) = sqrt(sum(abs(Phix - param_data.y).^2,'all')) ;
l1reg(1) = sum(abs(Psix),'all') ;
smooth_max(1) = max(abs(MbarxS(:))) ;
l2smooth(1) = sqrt(sum(abs( MxS - param_struct.l2_mean ).^2,'all' ) ) ;
time_tot(1) = 0 ;

%% display
    disp(' ')
    disp('*********************************************')
    disp('INITIALISATION')
    disp('*********************************************')
    disp(['d(xC, xS)                 = ',num2str(dist2(1))])
    disp('---------------------------------------------')
    disp(['|| Phi(xC) - y ||_2       = ',num2str(l2data(1))])
    disp(['      vs. l2 bound        = ',num2str(param_data.data_eps)])
    disp(['|| Psi(xC) ||_1           = ',num2str(l1reg(1))])
    disp(['      vs. l1 bound        = ',num2str(param_hpd.HPDconstraint)])
    disp('---------------------------------------------')
    disp(['|| MxS - mean ||_2        = ',num2str(l2smooth(1))])
    disp(['      vs. energy bound    = ',num2str(param_struct.l2_bound)])
    disp(['max( MxS - LMxS )       = ',num2str(smooth_max(1))])
    disp(['      vs. smooth tol.   = ',num2str(param_struct.tol_smooth)])
    disp('*********************************************')


%% ITERATIONS

for it = 1:param_algo.NbIt
    
    tic;
    v1old = v1 ;
    v2old = v2 ;
    v3old = v3 ;
    v4old = v4 ;
    
    v1_ = v1 + sigma1 * Phix ;
    v1 = v1_ - sigma1 * proj_l2ball( sigma1^(-1) * v1_, param_data.data_eps, param_data.y ) ;
    
    v2_ = v2 + sigma2 * Psix ;
    v2 = v2_ - sigma2 * real(oneProjector( sigma2^(-1) * v2_, param_hpd.HPDconstraint )) ;
    
    v3_ = v3 + sigma3 * MbarxS ;
% % %     v3 = v3_ - sigma3 * proj_l2ball( sigma3^(-1) * v3_, param_smooth.l2b_Mask, 0 );
    v3 = v3_ - sigma3 * min( max( sigma3^(-1) * v3_, -param_struct.tol_smooth), param_struct.tol_smooth) ;
    
    v4_ = v4 + sigma4 * MxS ;
    v4 = v4_ - sigma4 * proj_l2ball( sigma4^(-1) * v4_, param_struct.l2_bound, param_struct.l2_mean ) ;
    
    xC = xC - tau * ( (xC-xS) + real(param_data.Phit(2*v1-v1old)) + param_hpd.Psi(2*v2-v2old) ) ;
    xC = max(xC,0) ;
    xS = xS - tau * ( (xS-xC) + Mbaropt(2*v3-v3old) + Mopt(2*v4-v4old) ) ;
    xS = max(xS,0) ; %xS(param_smooth.back==1) = 0 ;
    
    Phix = param_data.Phi(xC) ;
    Psix = param_hpd.Psit(xC) ;
    MbarxS = Mbarop(xS) ;
    MxS = Mop(xS) ;
   
    %% critere
    time_tot(it+1) = toc;
    dist2(it+1) = sum(abs(xS(:) - xC(:)).^2) ;
    l2data(it+1) = sqrt(sum(abs(Phix - param_data.y).^2,'all') ) ;
    l1reg(it+1) = sum(abs(Psix),'all') ;
    smooth_max(it+1) = max(abs(MbarxS(:)));
    l2smooth(it+1) = sqrt(sum(abs( MxS - param_struct.l2_mean ).^2,'all')) ;
    
    %% display
    if mod(it, param_algo.display) == 0
        disp(' ')
        disp('*********************************************')
        disp(['it : ',num2str(it)])
        disp(['comp. time : ',num2str(sum(time_tot))])
        disp('*********************************************')
        disp(['d(xC, xS)               = ',num2str(dist2(it+1))])
        disp('---------------------------------------------')
        disp(['|| Phi(xC) - y ||_2     = ',num2str(l2data(it+1))])
        disp(['      vs. l2 bound      = ',num2str(param_data.data_eps)])
        disp(['|| Psi(xC) ||_1         = ',num2str(l1reg(it+1))])
        disp(['      vs. l1 bound      = ',num2str(param_hpd.HPDconstraint)])
        disp('---------------------------------------------')
        disp(['|| MxS - mean ||_2      = ',num2str(l2smooth(it+1))])
        disp(['      vs. energy bound  = ',num2str(param_struct.l2_bound)])
        disp(['max( MxS - LMxS )       = ',num2str(smooth_max(it+1))])
        disp(['      vs. smooth tol.   = ',num2str(param_struct.tol_smooth)])
        disp('*********************************************')
        
        h1 = figure(200);
        h1.WindowState = 'minimized';
        subplot 121, imagesc((xC)), axis image; colorbar, colormap gray
        xlabel(['xC - it = ',num2str(it)])
        subplot 122, imagesc((xS)), axis image; colorbar, colormap gray
        xlabel(['xS - it = ',num2str(it)])
        pause(0.1)
        set(0,'CurrentFigure',h1)
    end
    
    %% STOPPPPP
    cond_dist = abs(sqrt(dist2(it+1))-sqrt(dist2(it))) / sqrt(dist2(it+1)) ;
    cond_l2data = (1- 1e-5) *l2data(it+1)  ;
    cond_l2data_var = abs(l2data(it+1) - l2data(it)) / l2data(it+1) ;
    cond_l1reg = (1- 1e-5) *l1reg(it+1)  ;
    cond_l1reg_var = abs(l1reg(it+1) - l1reg(it)) / l1reg(it+1) ;
    cond_smooth_norm = (1-1e-5) * smooth_max(it+1) ;
    if it>10 ...
            && cond_dist < param_algo.stop_dist ...
            && cond_l2data_var < param_algo.stop_norm2 ...
            && cond_l1reg_var < param_algo.stop_norm2 ...
            && cond_l2data < param_data.data_eps ...
            && cond_l1reg < param_hpd.HPDconstraint ...
            && l2smooth(it+1) < param_struct.l2_bound ...
            && cond_smooth_norm < param_struct.tol_smooth
        disp(['STOP ITERATIONS - it = ',num2str(it)])
        break;
    end
end

%% RESULTATS

results.xC = xC ;
results.xS = xS ;

results.time = time_tot ;
results.dist2 = dist2 ;
results.l2data = l2data ;
results.l1reg = l1reg ;
results.l2smooth = l2smooth ;
results.smooth_max = smooth_max ;

end


%% FUNCTIONS



function p = proj_l2ball(x, eps, y)
% projection of x onto the l2 ball centered in y with radius eps
p = x-y ;
p = p* min(eps/norm(p(:)),1) ;
p = p+y ;

end