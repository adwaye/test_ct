function x = bwd_op(y, geom, mask)
%Y = zeros(size(mask));
%Y(mask>0) = y;
xhat = CTbeamAdj(y,geom.proj,geom.vol);
x    = xhat.*mask;
end