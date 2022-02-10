function [grad, gradt] = gradient_op(im)
dim_x = size(im,1);
dim_y = size(im,2);
% [x y] = meshgrid(1:1:dim_x,1:1:dim_y);
gradt = @(du) my_div(du,dim_x,dim_y);
grad  = @(u) my_grad(u) ;
end




function du = my_grad(u)
[Fx, Fy]=gradient(u);
Fx(1,:) = 0;
Fx(:,1) = 0;
Fy(1,:) = 0;
Fy(:,1) = 0;
du = cat(3,Fx,Fy);
end

function u = my_div(du,dim_x,dim_y)
Fx    = du(:,:,1);
Fy    = du(:,:,2);
[x, y] = meshgrid(1:1:dim_x,1:1:dim_y);
u     = -divergence(x,y,Fx,Fy);
end