function x = CTbeamAdj(y,W)
x = W'*y(:);
end

% % FP/BP wrapper function
% function Y = astra_wrap(X,T)
%   if strcmp(T, 'notransp')
%     % X is passed as a vector. Reshape it into an image.
%     [sid, s] = astra_create_sino_cuda(reshape(X,[vol_geom_.GridRowCount vol_geom_.GridColCount])', proj_geom_, vol_geom_);
%     astra_mex_data2d('delete', sid);
%     % now s is the sinogram. Reshape it back into a vector
%     Y = reshape(s',[numel(s) 1]);
%   else
%     % X is passed as a vector. Reshape it into a sinogram.
%     v = astra_create_backprojection_cuda(reshape(X, [proj_geom_.DetectorCount size(proj_geom_.ProjectionAngles,2)])', proj_geom_, vol_geom_);
%     % now v is the resulting volume. Reshape it back into a vector
%     Y = reshape(v',[numel(v) 1]);
%   end
% end
% %
% %astra_create_backprojection_cuda(reshape(X, [proj_geom.DetectorCount size(proj_geom.ProjectionAngles,2)])', proj_geom, vol_geom);
% %https://github.com/astra-toolbox/astra-toolbox/blob/master/samples/matlab/s015_fp_bp.m
% %try matlab radon to see if it looks the same
% vol_geom_ = vol_geom;
% proj_geom_ = proj_geom;
% b = reshape(y',[numel(y) 1]);
% Y = lsqr(@astra_wrap,b,1e-4,25);
% x = reshape(Y,[vol_geom.GridRowCount vol_geom.GridColCount]);