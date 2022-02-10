function y = my_grad(x)
[Fx Fy]=mygrad(x);

y = cat(3,Fx,Fy);
end