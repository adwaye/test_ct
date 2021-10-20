function [x, fid, reg, norm_it, snr_it] = MAP_primal_dual(param_data, param_map, tau, sig1, sig2, max_it, stop_it, stop_norm, SNR)
%% Primal-dual algorithm to minimise
%  lambda * || Psit(x) ||_1 s.t.  || Phi(x)-y ||_2 <= data_eps and x>=0

%% Initialisation
x =  param_data.Phit(param_data.y) ;
u = 0 * param_map.Psit(x) ;
v = 0 * param_data.y ;



Psitx = param_map.Psit(x) ;
Phix = param_data.Phi(x) ;

reg(1) = param_map.lambda * sum(abs(Psitx(:))) ;
fid(1) = sum(abs(Phix(:)-param_data.y(:)).^2) ;

%% Iterations

for it = 1:max_it
    xprev = x ;
    Psitx_ = Psitx ;
    Phix_ = Phix ;
    
    x = x - tau * (param_map.Psi(u) + param_data.Phit(v)) ;
    x(x<0) = 0 ;
    Psitx = param_map.Psit(x) ;
    Phix = param_data.Phi(x) ;

    u_ = u + sig1 * (2*Psitx - Psitx_) ;
    u = u_ - sig1 * prox_l1(u_/sig1, param_map.lambda/sig1) ;
    
    v_ = v + sig2 * (2*Phix - Phix_) ;
    v = v_ - sig2 * proj_l2ball(v_/sig2, param_data.data_eps, param_data.y) ;
    
    reg(it+1) = param_map.lambda * sum(abs(Psitx(:))) ;
    fid(it+1) = sum(abs(Phix(:)-param_data.y(:)).^2) ;
    
    norm_it(it) = norm(x(:) - xprev(:))/norm(x(:)) ;
    snr_it(it) = SNR(x) ;
    
    if mod(it,500)==0
        disp('**************************************')
        disp(['it = ', num2str(it)])
        disp(['SNR = ', num2str(snr_it(it))])
        disp(['l1 norm = ', num2str(reg(it+1))])
        disp(['data norm = ', num2str(fid(it+1))])
        disp(['   vs eps = ', num2str(param_data.data_eps)])
        disp(' ')
        disp(['rel. norm it.: ', num2str(norm_it(it))])
        figure(100)
        subplot 131, plot(reg), xlabel('it'), ylabel('l1 norm')
        subplot 132, hold off, plot(fid), hold on, plot(param_data.data_eps*ones(size(fid)),'r'), xlabel('it'), ylabel('l2 norm data')
        subplot 133, imagesc(x), axis image, colormap gray, colorbar, xlabel(['it=', num2str(it)])
        pause(0.1)
    end
    
    
    if it>50 ...
            && norm_it(it) < stop_norm ...
            && fid(it+1) <= param_data.data_eps * (1+stop_it) ...
            && abs(reg(it+1)-reg(it))/reg(it+1) < stop_it
        disp(['stop it = ', num2str(it)])
        break
    end
    
    
    
end


end


%% Proximity operators
function p = prox_l1(z, T) 
p = sign(z).*max(abs(z)-T, 0);
end

function p = proj_l2ball(x, eps, y)
% projection of x onto the l2 ball centered in y with radius eps
p = x-y ;
p = p* min(eps/norm(p(:)),1) ;
p = p+y ;

end