function y = fwd_op(x, geom,mask)

%Y(mask>0) = y;
y = CTbeam(mask.*x,geom.proj,geom.vol);
%y    = xhat(mask>0);
end