function [Psi, Psit] = wavelet_op(im, nlevel)

[C,S]=wavedec2(im,nlevel,'db8');   
Psit = @(x) wavedec2(x,nlevel,'db8')' ;
Psi  = @(x) waverec2(x,S,'db8') ;
end