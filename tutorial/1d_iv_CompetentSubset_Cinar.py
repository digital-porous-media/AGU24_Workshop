# Selecting a competent subset for visualization
# cinarturhan@utexas.edu

# Packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')
import skimage
from copy import deepcopy
import scipy
import os
import cc3d
import tifffile
os.chdir('../')
from src.vis_utils import plot_isosurface, orthogonal_slices
# os.chdir('C:/Users/cinar/Desktop/Github/dpm_tools')
# from dpm_tools.io import Image, Vector, ImageFromFile, read_image
# from dpm_tools.visualization.plot_2d import hist, plot_slice, make_gif
# from dpm_tools.visualization.plot_3d import orthogonal_slices, plot_isosurface, plot_streamlines
# from dpm_tools.visualization.plot_3d import plot_glyph, bounding_box

# Main Function
def find_porosity_visualization_interval(data, cube_size = 100, batch=100):
    '''
    Finds the best cubic interval for visaulizing the segmented dataset.
    
    data: Vector class from DRP, Image class from DRP, or 3D numpy array.
    cube_size: Size of the visalization cube, default is 100 (100x100x100). 
    batch: Batch over which to calculate the stats, default is 100.
    
    '''
    
    if str(type(data))=="<class 'dpm_tools.io.read_data.Vector'>":
        scalar_data = deepcopy(data.image)
    elif str(type(data))=="<class 'dpm_tools.io.read_data.Image'>":
        scalar_data = deepcopy(data)
    else:
        scalar_data = deepcopy(data)
    
    scalar_data[scalar_data==0]=199
    scalar_data[scalar_data!=199]=0
    scalar_data[scalar_data==199]=1
    
    size = scalar_data.shape[0]*scalar_data.shape[1]*scalar_data.shape[2]
    porosity = (scalar_data==1).sum()/size

    sample_size = cube_size

    # Inner cube increment
    inc = sample_size-int(sample_size*0.5)

    # One dimension of the given vector sample cube.
    max_dim = len(scalar_data)    

    batch_for_stats = max_dim-sample_size # Max possible batch number

    # Or overwrite:
    batch_for_stats = batch

    stats_array=np.zeros(shape=(5,batch_for_stats))

    i=0
    while i<batch_for_stats:
        mini = np.random.randint(low=0, high=max_dim-sample_size)
        maxi = mini+sample_size

        scalar_boot = scalar_data[mini:maxi,mini:maxi,mini:maxi]
        scalar_boot_inner = scalar_data[mini+inc:maxi-inc,mini+inc:maxi-inc,mini+inc:maxi-inc]

        scalar_boot_flat = scalar_boot.ravel()
        scalar_boot_inner_flat = scalar_boot_inner.ravel()

        labels_out_outside, N = cc3d.largest_k(
            scalar_boot, k=1, 
            connectivity=26, delta=0,
            return_N=True,
        )

        index_outside,counts_outside = np.unique(labels_out_outside,return_counts=True)
        counts_outside_sum = np.sum(counts_outside[1:])

        labels_out_inside, N = cc3d.largest_k(
            scalar_boot_inner, k=1, 
            connectivity=26, delta=0,
            return_N=True,
        )

        index_inside,counts_inside = np.unique(labels_out_inside,return_counts=True)
        counts_inside_sum = np.sum(counts_inside[1:])

        porosity_selected = (scalar_boot==1).sum()/sample_size**3

        if (porosity_selected<=porosity*1.2)&(porosity_selected>=porosity*0.8):
            stats_array[0,i] = counts_outside_sum
            stats_array[1,i] = counts_inside_sum     
            stats_array[2,i] = porosity_selected   
            stats_array[3,i] = mini
            stats_array[4,i] = scipy.stats.hmean([stats_array[0,i],
                                                  stats_array[1,i]])
            i+=1

        else:
            continue


    best_index = np.argmax(stats_array[4,:])
    best_interval = int(stats_array[3,best_index])

    print(f'Original Porosity: {round(porosity*100,2)} %\n' +
          f'Subset Porosity: {round(stats_array[2,best_index]*100,2)} %\n' +
          f'Competent Interval: [{best_interval}:{best_interval+cube_size},' +
          f'{best_interval}:{best_interval+cube_size},{best_interval}:{best_interval+cube_size}]')
    
    best_interval = (int(best_interval),int(best_interval+cube_size))
    
    return best_interval, stats_array


##############################################################

# Loading the datasets and visualizing them in 2D:
    
# os.chdir('C:/Users/cinar/Desktop/Research_Cinar_Offline/DRP Related/Codes/DPM Tutorial Data')

# TODO: Do we want to run these on the entire 512? or just the 256?
segmented_lrc32 = tifffile.imread('data/sandpack.tif')
segmented_Gambier = tifffile.imread('data/mtgambier.tif')
segmented_bead_pack_512= tifffile.imread('data/beadpack.tif')
segmented_castle_512 = tifffile.imread('data/castlegate.tif')
# segmented_lrc32 = np.fromfile('segmented_lrc32_512.ubc', dtype='uint8')
# segmented_lrc32 = segmented_lrc32.reshape((512,512,512))
segmented_lrc32_Image_256 = segmented_lrc32[0:256, 0:256, 0:256]
#
# segmented_Gambier = np.fromfile('segmented_Gambier_512.ubc', dtype='uint8')
# segmented_Gambier = segmented_Gambier.reshape((512,512,512))
segmented_Gambier_Image_256 = segmented_Gambier[0:256, 0:256, 0:256]
#
# segmented_bead_pack_512 = np.fromfile('segmented_bead_pack_512.ubc', dtype='uint8')
# segmented_bead_pack_512 = segmented_bead_pack_512.reshape((512,512,512))
segmented_bead_pack_512_Image_256 = segmented_bead_pack_512[0:256, 0:256, 0:256]
#
# segmented_castle_512 = np.fromfile('segmented_castle_512.ubc', dtype='uint8')
# segmented_castle_512 = segmented_castle_512.reshape((512,512,512))
segmented_castle_512_Image_256 = segmented_castle_512[0:256, 0:256, 0:256]

fig, ax = plt.subplots(nrows=2, ncols=2,figsize=(8,8))
ax = ax.flatten()

skimage.io.imshow(segmented_lrc32[0,:,:], cmap='gray', ax=ax[0])
ax[0].set_title('LRC32 Sand Pack',fontsize=14)

skimage.io.imshow(segmented_Gambier[0,:,:], cmap='gray', ax=ax[1])
ax[1].set_title('Gambier Limestone',fontsize=14)

skimage.io.imshow(segmented_bead_pack_512[0,:,:], cmap='gray', ax=ax[2])
ax[2].set_title('Glass Bead Pack',fontsize=14)

skimage.io.imshow(segmented_castle_512[0,:,:], cmap='gray', ax=ax[3])
ax[3].set_title('Castlegate Sandstone',fontsize=14)

plt.show()

# # Visualizing them in 3D:
#
fig_contours = plot_isosurface(segmented_lrc32_Image_256,show_isosurface=[0.5])
fig_contours.show()

fig_contours = plot_isosurface(segmented_Gambier_Image_256,show_isosurface=[0.5])
fig_contours.show()

fig_contours = plot_isosurface(segmented_bead_pack_512_Image_256,show_isosurface=[0.5])
fig_contours.show()

fig_contours = plot_isosurface(segmented_castle_512_Image_256,show_isosurface=[0.5])
fig_contours.show()

fig_orthogonal = orthogonal_slices(segmented_lrc32_Image_256,slider=True)
fig_orthogonal.show()

fig_orthogonal = orthogonal_slices(segmented_Gambier_Image_256,slider=True)
fig_orthogonal.show()

fig_orthogonal = orthogonal_slices(segmented_bead_pack_512_Image_256,slider=True)
fig_orthogonal.show()

fig_orthogonal = orthogonal_slices(segmented_castle_512_Image_256,slider=True)
fig_orthogonal.show()
##############################################################

# Finding the competent intervals
interval1, _ = find_porosity_visualization_interval(segmented_lrc32, cube_size=256, batch=100)
interval2, _ = find_porosity_visualization_interval(segmented_Gambier, cube_size=256, batch=100)
interval3, _ = find_porosity_visualization_interval(segmented_bead_pack_512, cube_size=256, batch=100)
interval4, _ = find_porosity_visualization_interval(segmented_castle_512, cube_size=256, batch=100)

##############################################################

# Selecting and 3D visualizing the intervals

segmented_lrc32_Image_competent = segmented_lrc32[109:365, 109:365, 109:365]
segmented_Gambier_Image_competent = segmented_Gambier[33:289, 33:289, 33:289]
segmented_bead_pack_512_Image_competent = segmented_bead_pack_512[111:367, 111:367, 111:367]
segmented_castle_512_Image_competent = segmented_castle_512[170:426, 170:426, 170:426]

fig_contours = plot_isosurface(segmented_lrc32_Image_256, show_isosurface=[0.5])
fig_contours.show()

fig_contours = plot_isosurface(segmented_Gambier_Image_256, show_isosurface=[0.5])
fig_contours.show()

fig_contours = plot_isosurface(segmented_bead_pack_512_Image_256, show_isosurface=[0.5])
fig_contours.show()

fig_contours = plot_isosurface(segmented_castle_512_Image_256, show_isosurface=[0.5])
fig_contours.show()

fig_orthogonal = orthogonal_slices(segmented_lrc32_Image_256, slider=True)
fig_orthogonal.show()

fig_orthogonal = orthogonal_slices(segmented_Gambier_Image_256, slider=True)
fig_orthogonal.show()

fig_orthogonal = orthogonal_slices(segmented_bead_pack_512_Image_256, slider=True)
fig_orthogonal.show()

fig_orthogonal = orthogonal_slices(segmented_castle_512_Image_256, slider=True)
fig_orthogonal.show()
##############################################################

# Case where selection makes differences, here we use 128^3 cubes, and since it
# is smaller compared to the main sample (512^3), the selection can result in
# a misleading visual compared to the original.


# Randomly selecting a subset:
segmented_Gambier_Image_128 = segmented_Gambier[128:256, 128:256, 128:256]
porosity = np.sum(segmented_Gambier_Image_128 == 0)/(128**3)*100
print(round(porosity, 1), '%')

fig_contours = plot_isosurface(segmented_Gambier_Image_128, show_isosurface=[0.5])
fig_contours.show()

fig_orthogonal = orthogonal_slices(segmented_Gambier_Image_128, slider=True)
fig_orthogonal.show()


# Finding a competent subset:
interval, _ = find_porosity_visualization_interval(segmented_Gambier, cube_size=128, batch=300)

segmented_Gambier_Image_competent_128 = segmented_Gambier[51:179, 51:179, 51:179]


fig_contours = plot_isosurface(segmented_Gambier_Image_competent_128,show_isosurface=[0.5])
fig_contours.show()

fig_orthogonal = orthogonal_slices(segmented_Gambier_Image_competent_128,slider=True)
fig_orthogonal.show()


