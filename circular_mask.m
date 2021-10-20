function array = circular_mask(nx,ny)
array = zeros(nx,ny);
center = [nx/2,ny/2];
radius = nx/2;
for i = 1:nx
    for j = 1:ny
        if (i-center(1))^2+(j-center(2))^2<radius^2
            array(i,j)=1;
        end
    end 
end
end