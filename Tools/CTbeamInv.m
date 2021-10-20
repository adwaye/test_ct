function x = CTbeamInv(y,proj_geom,vol_geom)
%
%astra_create_backprojection_cuda(reshape(X, [proj_geom.DetectorCount size(proj_geom.ProjectionAngles,2)])', proj_geom, vol_geom);
%https://github.com/astra-toolbox/astra-toolbox/blob/master/samples/matlab/s015_fp_bp.m
%try matlab radon to see if it looks the same
rec_id = astra_mex_data2d('create','-vol',vol_geom);
cfg = astra_struct('SIRT_CUDA');
cfg.ReconstructionDataId = rec_id;
sinogram_id = astra_mex_data2d('create','-sino',proj_geom,y);
cfg.ProjectionDataId = sinogram_id;
% Create the algorithm object from the configuration structure
alg_id = astra_mex_algorithm('create', cfg);

% Run 150 iterations of the algorithm
astra_mex_algorithm('iterate', alg_id, 150);

% Get the result
rec = astra_mex_data2d('get', rec_id);
x = rec;
astra_mex_algorithm('delete', alg_id);
astra_mex_data2d('delete', rec_id);
astra_mex_data2d('delete', sinogram_id)
end