function test_projection
clear all
%clc
close all


addpath Algos/
addpath Data/
addpath Tools/
addpath(genpath('Tools'))
P = phantom(256);
%load("/home/adwaye/PycharmProjects/CT-UQ/CTimages.mat");
%P = gtruth*5;
nx=size(P,1);
ny=size(P,2);


%we will create a linear opeator that will create a sinogram and find its
%adjoint
vol_geom = astra_create_vol_geom(nx,ny);
%create a parallel projection geometry with spacing =1 pixel, 384 
geom.spacing =1.0;% 1/(nx*ny);
geom.ndetectors = 384;
geom.angles     =  linspace2(0,pi,200);
proj_geom = astra_create_proj_geom('parallel', geom.spacing, geom.ndetectors, geom.angles);
geom.proj  = proj_geom;
geom.vol = vol_geom;
stddev   = 1e-4;
%



%y = CTbeam(P,geom.proj,geom.vol);
mask = circular_mask(nx,ny);
phi = @(x) CTbeam(x,proj_geom,vol_geom) ;
%astra_fun = @(v,T) astra_wrap(v,T,vol_geom,proj_geom);
phit = @(x) lsqr(@astra_fun,reshape(x',[numel(x) 1]),1e-4,500);




%volume = astra_create_backprojection_cuda(y, proj_geom, vol_geom);

%y = phi(P.*mask);
y = phi(P);
recon = phit(y);
%recon = CTbeamAdj(y,geom.proj,geom.vol);
y_scaled = rescale(y);
Mx = size(y,1);My = size(y,2);
noise = (randn(Mx,My)) * stddev/sqrt(2);
noisy_data = y+noise;
%Noise = poissrnd(I1);
%recon_noisy = phit(noisy_data);
figure(1); imshow(P, []);colorbar();title("Phantom");
figure(2); imshow(y, []);colorbar();title("sinogram");
figure(3);imshow(recon,[]);colorbar();title("adjoint of phi");
figure(4);imshow(noisy_data,[]);colorbar();title("noisy data");
%figure(5);imshow(recon_noisy,[]);colorbar();title("Reconstructed from noisy data");
xtmp  = rand(ny,nx) ;
ytmp  = phi(rand(ny,nx)) ;
Pxtmp = phi(xtmp) ;
Ptytmp = phit(ytmp) ;
fwd = real(Pxtmp(:)'*ytmp(:)) ;
bwd = xtmp(:)'* Ptytmp(:) ;
disp("size image")
size(xtmp)
disp("size mapped image")
size(ytmp)
disp('test adjoint operator')
disp(['fwd = ', num2str(fwd)])
disp(['bwd = ', num2str(bwd)])
disp(['diff = ', num2str(norm(fwd-bwd)/norm(fwd))])

normPhi = op_norm2(phi, phit, [ny, nx], 1e-4, 200, 0);    
disp(['norm of phi = ', num2str(normPhi)])









function Y = astra_fun(X,T)
  if size(X,2)>1
      X = reshape(X',[numel(X) 1]);
  end
  if strcmp(T, 'notransp')
    % X is passed as a vector. Reshape it into an image.
    [sid, s] = astra_create_sino_cuda(reshape(X,[vol_geom.GridRowCount vol_geom.GridColCount])', proj_geom, vol_geom);
    astra_mex_data2d('delete', sid);
    % now s is the sinogram. Reshape it back into a vector
    Y = reshape(s',[numel(s) 1]);
  else
    % X is passed as a vector. Reshape it into a sinogram.
    v = astra_create_backprojection_cuda(reshape(X, [proj_geom.DetectorCount size(proj_geom.ProjectionAngles,2)])', proj_geom, vol_geom);
    % now v is the resulting volume. Reshape it back into a vector
    Y = reshape(v',[numel(v) 1]);
  end
end




function val = op_norm2(A, At, im_size, tol, max_iter, verbose)
%% computes the maximum eigen value of the compund operator At*A
x = randn(im_size);
x = x/norm(x(:));
init_val = 1;

for k = 1:max_iter
    y_ = A(x);
    x = At(y_);
    x = reshape(x,im_size);
    val = norm(x(:));
    rel_var = abs(val-init_val)/init_val;
    if (verbose > 1)
        fprintf('Iter = %i, norm = %e \n',k,val);
    end
    if (rel_var < tol)
       break;
    end
    init_val = val;
    x = x/val;
    
end

if (verbose > 0)
    fprintf('Norm = %e \n\n', val);
end

end

end



