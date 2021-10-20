function [sampling_array_fft,sampling_array] = radial_sampling_op(im,num_angles)
block  = 0;
im_shp = size(im);
nx     = im_shp(1);
ny     = im_shp(2);
sampling_array = zeros(nx, ny);
angles = linspace(0, 2*pi, num_angles);
angles = angles(1:num_angles-1);
center = [fix(nx/2),fix(ny/2)];

for r=1:fix(sqrt(2)*fix(max(im_shp)/2))
   for i=1:num_angles-1
       phi = angles(i);
       idx = ceil([r*cos(phi),r*sin(phi)]);
       
       glob_idx1 = center(1) + idx(1);
       glob_idx2 = center(2) + idx(2);
       if 0 < glob_idx1 & glob_idx1 <= im_shp(1) & 0 < glob_idx2 & glob_idx2 <= im_shp(2)
           %disp('entering if')
           %disp(glob_idx1)
           %disp(glob_idx2)
           sampling_array(glob_idx1,glob_idx2)=1;
       end
      
   end 
end

sampling_array(center(1)-fix(block/2)+1:center(1)+fix(block/2)+1,center(2) - fix(block/2) +1: center(2)+fix(block/2)+1)=1;
sampling_array_fft = fftshift(sampling_array);

end