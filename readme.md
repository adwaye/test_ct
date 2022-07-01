# Data
## Raw Data
1. The raw CT scans can be found under <tt>Data/ct_scans</tt>. There are two folders, each corresponding to two different CT scans. 
2. Masks for artefacts can be found under <tt>artcon_DE_CTPA_1_5_Bv38_2_F_0.nrrd</tt> or <tt>pe_DE_CTPA_1_5_Bv38_2_F_0.nrrd</tt> .
3. The raw CT scan is saved under <tt>DE_CTPA_1_5_Bv38_2_F_0.nrrd</tt>
4. To convert the <tt>.nrrd</tt> files to <tt>.mat</tt> one can run the script called <tt>nrrd_to_mat_files.py</tt>. 
5. To analyse the data, the ct scans need to be converted to 2d slices, which can be done by running <tt>nrrd_to_mat_files.py</tt>.

## Slice Data
1. The slice data can be found in <tt>Data/ct_scans/ct1</tt>. 
2. They are saved as <tt>.mat</tt> files with fields <tt>CT</tt> and <tt>labels</tt>
3. The labels are 2d arrays that are zero where the structure is not present in the image. They need to be converetd to a binary mask.


# Code
## Matlab Code
### <tt>experiments_gradM_M.m</tt>

The main file is called <tt>experiments_gradM_M.m</tt>. The input is a slice data <tt>.mat</tt> file that have fields <tt>CT</tt>, <tt>texture_mask</tt> and <tt>labels</tt> with both arrays being the same size. These can be found under 
 <tt>Data/ct_scans/ct1</tt>. These were made from the slices created by running <tt>nrrd_to_mat_files.py</tt> by adding a new field called <tt>texture_mask</tt>. This represents an area in the image whose pixel intensity should have the same distribution as the pixel intensity inside the area where the arteafact is found. <tt>texture_mask</tt> is used to set the radii of the balls that define the set $\mathcal{S}$.


Running <tt>experiments_gradM_M.m</tt> will iterate through a user defined list of detector setups (different noise and number of angles), radii for the balls defining the set $\mathcal{S}$. These are defined by the quantiles of the pixel distribution at <tt>texture_mask</tt>. The results of the forward map simulation for each detector/noise set up is saved in the <tt>results_path</tt> created in <tt>experiments_gradM_M.m</tt> as well as the resuls of the BUQO simulation for each detector setup and pixel quantile used to define the radii of the balls definning the setv $\mathcal{S}$. The file names will reflect the setups used.

The result_path will be found under <tt>Figures</tt> and will be named after the types of norm used to define the balls bounding the energy of the gradient and intensity of the image within the artefact location

### <tt>inspect_xmap.m</tt>
This script will display the map estimate for the specified slice with name <tt>slice_name</tt> which was simulated using noise levels and number of angles that are specified by the user in the scipt (line 43 at writing this guide).

The user then has the choice to define a texture_mask and an artefact mask by adding an if statement as the ones present as from line 79, and saving a <tt>.mat</tt> file with entries
'xmap','fid','reg','norm_it','snr_it','time_it','time_total',...
                     'texture_mask','mask_struct','im_true','CT'
where the 'mask_struct' shows the location of the artefact the user detected in the xmap, and 'texture_mask' shows an area whose pixel intensity will drive the radii of the balls defininf the set $\mathcal{S}$. This is saved under <tt>Data/aretfacts</tt>


### <tt>test_artefact.m</tt>
This script uses the output from <tt>inspect_xmap.m</tt> to run BUQO on the artefacts whose location and sampled pixels are defined by <tt>mask_struct</tt> and <tt>texture_mask</tt> in the <tt>.mat</tt> files saved at <tt>Data/aretfacts</tt>. The output is saved under <tt>Figures/[folder_name]</tt> where [folder_name] has naming convention <tt>artefact_[Lx_gradM_Lx_M]</tt>.



## Python Code
### <tt>nrrd_to_mat_files.py</tt>

The <tt>main</tt> function reads all nrrd files in a location containing nrrd files specified in the [raw data section](##raw_data) and converts them into a <tt>.mat</tt>.

The loop specified in the file will iterate through slices of the specified raw CT scan and a corresponding  <tt>.nrrd</tt> files corresponding to a mask and save the label-image pair in a .mat file as 2d slices. 


### <tt>Experiments.ipynb</tt>

This is a jupyter notebook that reads the output of the BUQO algorithm from the specified folder and converts the metrics and results in a panadas dataframe. The pandas dataframe can then be used to plot the different results for publication/presentations.







