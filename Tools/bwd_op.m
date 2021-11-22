function x = bwd_op(y, W, mask)
%Y = zeros(size(mask));
%Y(mask>0) = y;
xhat = CTbeamAdj(y,W);
x    = reshape(xhat,[size(mask)]).*mask;
end