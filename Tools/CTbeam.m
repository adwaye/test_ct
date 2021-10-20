function y = CTbeam(x,proj_geom,vol_geom)
[id,sinogram]=astra_create_sino_gpu(x, proj_geom, vol_geom);
y = sinogram;
astra_mex_data2d('delete', id);
end