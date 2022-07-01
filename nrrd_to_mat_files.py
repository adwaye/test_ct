import nrrd,os,argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description="Save nrrd as mat files")
parser.add_argument("--src", type=str,default="/home/adwaye/matlab_projects/test_CT/Data/ct_scans/ct1"
                    ,help="location of the nrrd files t be converted")
parser.add_argument("--tgt", type=str,default="/home/adwaye/matlab_projects/test_CT/Data/ct_scans/ct1"
                    ,help="destination folder of mat files")



args = parser.parse_args()

def main():
    source_loc = args.src
    dest_loc   = args.tgt

    file_names = [os.path.join(source_loc,f) for f in os.listdir(source_loc) if f[-4:]=='nrrd']

    for f in file_names:
        name          = f.split('/')[-1].split('.')[0]
        nrrd_file     = nrrd.read(f)
        voxels        = nrrd_file[0]
        tgt_file_name = os.path.join(dest_loc,name+'.mat')
        sio.savemat(file_name=tgt_file_name,mdict={"array":voxels})




if __name__=="__main__":
    ct_name    = "ct1" #"ct1 or ct2
    label      = "pe" #"pe" or "artcon"
    _save      = True
    _plot      = True
    if ct_name == "ct1" and label=="pe":
        ystart = 274
        yend   = 274
        ystep   = 12
        xstart = 220
        xend   = 230
        xstep  = 1
        zstart = 177
        zend   = 177
        zstep  = 12
    elif ct_name == "ct1" and label=="artcon":
        xstart = 274
        xend   = 310
        xstep  = 10
        ystart = 211
        yend   = 241
        ystep  = 10
        zstart = 297
        zend   = 320
        zstep  = 6

    source_loc = "/home/adwaye/matlab_projects/test_CT/Data/ct_scans/ct1"
    file_names = [os.path.join(source_loc,f) for f in os.listdir(source_loc) if f[-4:] == 'nrrd']
    files      = [f.split('/')[-1] for f in file_names]

    lab_name = ''
    im_name  = ''
    for f in files:
        if f.split('_')[0]==label:
            lab_name = f
        if f.split('_')[0] == 'DE':
            im_name = f


    nrrd_file = nrrd.read(os.path.join(source_loc,im_name))
    ct_scan   = nrrd_file[0]
    nrrd_file = nrrd.read(os.path.join(source_loc,lab_name))
    labels    = nrrd_file[0]


    if _plot:
        fig,ax = plt.subplots(nrows=1,ncols=2)
        fig.suptitle("x-slice")
        ax[0].set_title("ct slice")
        ax[1].set_title("label slice")
    for x in range(xstart,xend,xstep):
        slice_img = ct_scan[x,:,:]
        slice_lab = labels[x,:,:]
        if _plot:
            ax[1].imshow(slice_lab)
            ax[0].imshow(slice_img)
            plt.pause(1)
            fig.savefig(os.path.join(source_loc,label + "_xslice_{:}.png".format(x)))
        if _save:
            sio.savemat(file_name=os.path.join(source_loc,label+"_xslice_{:}.mat".format(x)),
                    mdict={"labels":slice_lab,"CT":slice_img})
    if _plot:
        fig,ax = plt.subplots(nrows=1,ncols=2)
        fig.suptitle("y-slice")
        ax[0].set_title("ct slice")
        ax[1].set_title("label slice")

    for y in range(ystart,yend,ystep):
        slice_img = ct_scan[:,y,:]
        slice_lab = labels[:,y,:]
        if _plot:
            ax[1].imshow(slice_lab)
            ax[0].imshow(slice_img)
            plt.pause(1)
            fig.savefig(os.path.join(source_loc,label + "_yslice_{:}.png".format(y)))
        if _save:
            sio.savemat(file_name=os.path.join(source_loc,label+"_yslice_{:}.mat".format(y)),
                    mdict={"labels":slice_lab,"CT":slice_img})


    if _plot:
        fig,ax = plt.subplots(nrows=1,ncols=2)
        fig.suptitle("z-slice")
        ax[0].set_title("ct slice")
        ax[1].set_title("label slice")
    for z in range(zstart,zend,zstep):
        slice_img = ct_scan[:,:,z]
        slice_lab = labels[:,:,z]
        if _plot:
            ax[1].imshow(slice_lab)
            ax[0].imshow(slice_img)
            plt.pause(1)
            fig.savefig(os.path.join(source_loc,label + "_zslice_{:}.png".format(z)))
        if _save:
            sio.savemat(file_name=os.path.join(source_loc,label+"_zslice_{:}.mat".format(z)),
                    mdict={"labels":slice_lab,"CT":slice_img})



# make boundaries for each image
# import cv2
# ret,thresh = cv2.threshold(slice_lab,0,1,0)
# contours, hierarchy = cv2.findContours(thresh.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
# tmp = np.zeros_like(slice_lab).astype(np.uint8)
# boundary = cv2.drawContours(cv2.UMat(tmp), contours, -1, (255,255,255), 1)
# boundary[boundary > 0] = 255


