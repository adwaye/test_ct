function results = Project_to_S(xmap, param_algo, param_data,  param_struct)


%% Operateurs
% if ~isfield(param_struct, 'back'), param_struct.back = zeros(size(xmap_inp)) ; end;
if ~isfield(param_algo, 'display'), param_algo.display = 100 ; end;


Gradop =@(x) param_struct.Gradop(x) ;
Gradopt =@(v) param_struct.Gradopt(v);

Mask_op  = @(x) param_struct.Mask_op*x(:) ;
Mask_opt =@(v) reshape(param_struct.Mask_op'*v, param_data.Ny, param_data.Nx) ;
%% STEP-SIZES
% sigma1 = 1/sqrt(param_data.normPhi+param_struct.normPsi) ;
% sigma2 = sigma1 ;
% normMask = op_norm(Mask_op, Mask_opt, [param_data.Ny, param_data.Nx], 1e-4, 200, 0) ;
% sigma3 = 1 / sqrt( normMask ) ;
% normGrad = op_norm(Gradop, Gradopt, [param_data.Ny, param_data.Nx], 1e-4, 200, 0) ;
% if nargin ==6
% sigma4 = 1 / sqrt( normGrad ) ;
% disp("estimating sigma4 as reciprocal of sqrt of ||M||");
% end
% disp(['sigma4= ',num2str(sigma4)]);               
tau = 0.9 * 2*sqrt(2);

%% INITIALISATION
xS = 0 * xmap ;
u = 0 * Mask_op(xS) ;
v = 0 * Gradop(xS) ;

% MbarxS = Mbarop(xS) ;
MGradxS = Gradop(xS) ;
MxS     = Mask_op(xS);


%% critere
dist2(1) = sum(abs(xS(:) - xmap(:)).^2) ;
mask_energy(1) = sqrt(sum(abs( MxS - param_struct.l2_mean_pix ).^2,'all' ) ) ;
grad_energy(1) = sqrt(sum(abs( MGradxS(:) - param_struct.l2_mean_grad ).^2,'all' ) ) ;
time_tot(1)    = 0 ;

%% display
    disp(' ')
    disp('*********************************************')
    disp('INITIALISATION')
    disp('*********************************************')
    disp(['d(xmap, xS)                 = ',num2str(dist2(1))])
    disp('---------------------------------------------')
    disp(['|| MxS-mean_px ||_2                = ',num2str(mask_energy(1))])
    disp(['      vs. upper bound    = ',num2str(param_struct.l2_bound_pix)])
    disp('---------------------------------------------')
    disp(['|| grad (MxS)-mean_grad ||_2                = ',num2str(grad_energy(1))])
    disp(['      vs. upper bound    = ',num2str(param_struct.l2_bound_grad)])
    disp('*********************************************')


%% ITERATIONS

for it = 1:param_algo.NbIt
    
    tic;
    
    
    uold = u ;
    vold = v ;
    
    %condition on Mbar
    
    xS = xmap - 0.5*tau*(Mask_opt(u)*Gradopt(v));
    xS = max(xS,0) ;
%     xS(xS<0) = 0 ;
    
    u_ = u  - tau*MxS;
    u  = u_ - tau * proj_l2ball( tau^(-1) * u_, param_struct.l2_bound_pix, param_struct.l2_mean_pix ) ;
    
    
    %condition on grad
    v_ = v  - tau * MGradxS;
    v  = v_ - tau * proj_l2ball( tau^(-1) * v_, param_struct.l2_bound_grad, param_struct.l2_mean_grad ) ;
    
    %making xS

    
    
    MGradxS = Gradop(xS) ;
    MxS     = Mask_op(xS);
    
   
    %% critere
    
    dist2(it+1) = sum(abs(xS(:) - xmap(:)).^2) ;
    mask_energy(it+1) = sqrt(sum(( MxS(:) - param_struct.l2_mean_pix ).^2,'all' ) ) ;
    grad_energy(it+1) = sqrt(sum(( MGradxS(:) - param_struct.l2_mean_grad ).^2,'all' ) ) ;

    %% display
    if mod(it, param_algo.display) == 0
        disp(' ')
        disp('*********************************************')
        disp(['d(xmap, xS)                 = ',num2str(dist2(it+1))])
        disp('---------------------------------------------')
        disp(['|| MxS-mean_px ||_2                = ',num2str(mask_energy(it+1))])
        disp(['      vs. upper bound    = ',num2str(param_struct.l2_bound_pix)])
        disp('---------------------------------------------')
        disp(['|| grad (MxS)-mean_grad ||_2                = ',num2str(grad_energy(it+1))])
        disp(['      vs. upper bound    = ',num2str(param_struct.l2_bound_grad)])
        disp('*********************************************')
        
        h1 = figure(200);
%         h1.WindowState = 'minimized';
        subplot 121, imagesc((xmap)), axis image; colorbar, colormap gray
        xlabel(['xmap - it = ',num2str(it)])
        subplot 122, imagesc((xS)), axis image; colorbar, colormap gray
        xlabel(['xS - it = ',num2str(it)])
        pause(0.1)
        set(0,'CurrentFigure',h1)
    end
    
    %% STOPPPPP
    cond_dist = abs(sqrt(dist2(it+1))-sqrt(dist2(it))) / sqrt(dist2(it+1)) ;
    if it>10 ...
            && cond_dist < param_algo.stop_dist ...
            && mask_energy(it+1) < param_struct.l2_bound_pix ...
            && grad_energy(it+1) < param_struct.l2_bound_grad
        disp(['STOP ITERATIONS - it = ',num2str(it)])
        break;
    end
end

%% RESULTATS


results.xS = xS ;

results.time = time_tot ;
results.dist2 = dist2 ;
results.mask_energy = mask_energy ;
results.grad_energy = grad_energy ;
end


%% FUNCTIONS



function p = proj_l2ball(x, eps, y)
% projection of x onto the l2 ball centered in y with radius eps
p = x-y ;
p = p* min(eps/norm(p(:)),1) ;
p = p+y ;

end