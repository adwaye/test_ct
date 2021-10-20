function x = TFadj(y,N)
x = real(ifft2(ifftshift(y)))*sqrt(N) ;
end