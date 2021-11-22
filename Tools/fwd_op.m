function y = fwd_op(x,W,mask)

%Y(mask>0) = y;
yhat = CTbeam(mask.*x,W);
y = reshape(yhat,[W.proj_size]);
%y    = xhat(mask>0);
end