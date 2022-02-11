function [grad, gradt] = gradient_op(im,mask)
dim_x = size(im,1);
dim_y = size(im,2);
% [x y] = meshgrid(1:1:dim_x,1:1:dim_y);
% gradt = @(du) my_div_backward(du,dim_x,dim_y);
% grad  = @(u) my_grad_forward(u) ;
gradt = @(du) my_div_backward_masked(du,dim_x,dim_y,mask);
grad  = @(u) my_grad_forward_masked(u,mask) ;

end





function y = my_grad_forward_masked(u,mask)

mmask = cat(3,mask,mask);
du    = my_grad_forward(u);

y     =  du(mmask>0);
end


function x  = my_div_backward_masked(du,dim_x,dim_y,mask)
mmask = cat(3,mask,mask);
Y = zeros(size(mmask));
Y(mmask>0) = du;
x = my_div_backward(Y,dim_x,dim_y);
end


function du = my_grad_forward(u)
dim_x = size(u,1);
dim_y = size(u,2);
Fy = diff(u);
Fy(dim_x,:) = 0-u(dim_x,:);
Fx = diff(u');
Fx = Fx';
Fx(:,dim_y) = 0-u(:,dim_y);
% dim_x = size(u,1);
% dim_y = size(u,2);
% [Fx, Fy]=gradient(u);
% Fx(1,:) = (Fx(2,:)-0)/2;
% Fx(dim_x,:) = (0-Fx(dim_x-1,:))/2;
% Fy(:,1) = (Fy(:,2)-0)/2;
% Fy(:,dim_y) = (0-Fy(:,dim_y-1))/2;
du = cat(3,Fx,Fy);
end
% 
% 
% 
% 
function u = my_div_backward(du,dim_x,dim_y)
Fx    = du(:,:,1);
Fxx   = zeros(size(Fx'));
Fy    = du(:,:,2);
Fyy   = zeros(size(Fy));
Fx = Fx';
Fxx(2:dim_y,:) = diff(Fx);
Fxx(1,:) = Fx(1,:);
Fxx = Fxx';
Fyy(2:dim_x,:)   = diff(Fy);
Fyy(1,:) = Fy(1,:);
% [x, y] = meshgrid(1:1:dim_x,1:1:dim_y);
% u     = -divergence(x,y,Fx,Fy);
u = -Fxx-Fyy;
end



function du = my_grad_central(u)
dim_x = size(u,1);
dim_y = size(u,2);
[Fx, Fy]=gradient(u);
Fx(:,1) = (Fx(:,2)-0)/2.0;
Fx(:,dim_x) = (0-Fx(:,dim_x-1))/2.0;
Fy(1,:) = (Fy(2,:)-0)/2.0;
Fy(dim_y,:) = (0-Fy(dim_y-1,:))/2.0;
du = cat(3,Fx,Fy);
end

function u = my_div_central(du,dim_x,dim_y)
Fx    = du(:,:,1);
Fy    = du(:,:,2);
Fxx = gradient(Fx);
Fyy   = gradient(Fy');
% Fyy(1,:) = (Fyy(2,:)-0)/2;
% Fyy(dim_y,:) = (0-Fyy(dim_y-1,:))/2;
% disp(["|Fxy-Fyx|",num2str(norm(Fxy-Fyx))])
Fyy   = Fyy';
Fxx(:,1) = (Fxx(:,2)-0)/2.0;
Fxx(:,dim_x) = (0-Fxx(:,dim_x-1))/2.0;
Fyy(1,:) = (Fyy(2,:)-0)/2.0;
Fyy(dim_y,:) = (0-Fyy(dim_y-1,:))/2.0;
% [x, y] = meshgrid(1:1:dim_x,1:1:dim_y);
% u     = -divergence(x,y,Fx,Fy);
u = -Fxx-Fyy;
end

% _ADJ_METHOD = {'central': 'central',
%                'forward': 'backward',
%                'backward': 'forward'}
% 
% _ADJ_PADDING = {'constant': 'constant',
%                 'symmetric': 'symmetric_adjoint',
%                 'symmetric_adjoint': 'symmetric',
%                 'periodic': 'periodic',
%                 'order0': 'order0_adjoint',
%                 'order0_adjoint': 'order0',
%                 'order1': 'order1_adjoint',
%                 'order1_adjoint': 'order1',
%                 'order2': 'order2_adjoint',
%                 'order2_adjoint': 'order2'}