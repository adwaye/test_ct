function y = TF(x,N)
y = fftshift(fft2(x))/sqrt(N) ;
y(abs(y)<1e-10) = 0 ;
end