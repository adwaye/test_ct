function Y = astra_wrap(X,T,vol_geom,proj_geom)
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