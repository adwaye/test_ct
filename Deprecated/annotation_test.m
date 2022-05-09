filename = "Data/ct_scans/ct1/DE_CTPA_1_5_Bv38_2_F_0.mat";
img = load(filename);   
vol = img.array;

filename = "Data/ct_scans/ct1/pe_DE_CTPA_1_5_Bv38_2_F_0.mat";
img = load(filename);   
label = img.array;
mask = 2*uint16(label>0);


filename = "Data/ct_scans/ct1/artcon_DE_CTPA_1_5_Bv38_2_F_0.mat";
img = load(filename);   
label = img.array;
mask = uint16(label>0);





%pe extraction ct1
x = 245;y = 271;z=186;
slicedlab = squeeze(mask(:,:,1));
s
%contrast artefacts extraction ct1
x = 372;y = 284;z=317;
slicedlab = squeeze(255*mask(:,:,z));
slicedCT  = squeeze(255*vol(:,:,z));
[row col] = find(slicedlab)
imshow(slicedlab);