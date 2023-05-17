"""
Simple module for an ez clear pipeline for 1x mesoSPIM stacks so they are in the same orientation
as BrainSaw data. Then save a 25 micron downsampled stack.
Run from directory with sample data in it. It will process everything in the folder.
"""

from glob import glob
from skimage import io, filters, morphology, measure
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom # To resize the image
import mesospim_python_tools
import cv2
import SimpleITK as sitk
import re

dsRoot = 'downsampledStacks'
dsSub =  '025_micron'




def crop_full_stacks(stacks,do_plot=False):
    """
    Crops the full size stacks. At the moment this expects as input the background
    channel. If this has both lightsheets then both should be provided. No checks yet!

    Inputs
    stacks : Takes as input the structure returned by
        mesospim_python_tools.io.return_mesoSPIM_files_in_path

    We write to disk new stacks with 'cropped_' appended ot the file name.

    """

    if not isinstance(stacks,list):
        stacks = [stacks]

    # Load images associated with all stacks
    im_stacks = list()
    for t_stack in stacks:
        fname = t_stack['image_file_name']
        print('Loading %s' % fname)

        im_stacks.append(io.imread(fname))


    # Down-sample and merge
    ax_downsample = np.array([0.25,0.25,0.25])
    ds_stacks = [zoom(t_stack, ax_downsample, order=1) for t_stack in im_stacks]

    # Find the bounding boxes
    bboxes = get_crop_box(sum(ds_stacks), do_plot=do_plot)
    cropbox = [b[0].bbox for b in bboxes]


    # Pull out the indexes that we need and up-sample to crop box
    ax_0 = np.array([cropbox[1][0], cropbox[1][2]])
    ax_1 = np.array([cropbox[0][0], cropbox[0][2]])
    ax_2 = np.array([cropbox[0][1], cropbox[0][3]])


    # Scale up
    ax_0 = np.round(ax_0 * (1/0.25)).astype('uint16')
    ax_1 = np.round(ax_1 * (1/0.25)).astype('uint16')
    ax_2 = np.round(ax_2 * (1/0.25)).astype('uint16')


    # Apply the crop boxes to the full size data and save
    for (im,metadata) in zip(im_stacks,stacks):

        im = im[ax_0[0]:ax_0[1],
                ax_1[0]:ax_1[1],
                ax_2[0]:ax_2[1]]

        fname = metadata['image_file_name']
        save_fname = 'crop_'+fname
        print('Saving to %s' % save_fname)
        io.imsave(save_fname,im)

    return ds_stacks

def get_crop_box(im: np.ndarray, otsu_scale=0.5, do_plot=False) -> np.ndarray:
    """
    Gets a bounding box in all three axes to crop

    Inputs
    im : 3D stack that we will process
    otsu_scale : how much to scale otsu by
    do_plot : Bool (false by default) defining whether we will make plots showing
            what was done.

    Return
    crop_box : ndarray defining the crop box to apply to the image. This is a list of
            two tuples organised as minr, minc, maxr, maxc
            The first box corresponds to axes 1/2 and the second to 0/2
    """

    # First max project captures first two axes and the second does the third
    planes = [np.max(im,axis=0), np.max(im,axis=1)]

    # Otsu threshold with a reduction that has no good basis
    thresh = [filters.threshold_otsu(t_plane)*otsu_scale for t_plane in planes]

    # Use the threshold to make a black and white image
    bw = [t_plane>t_thresh for t_plane, t_thresh in zip(planes,thresh)]


    # Morph filter the bw image to get rid of small stuff
    filter_size = 2 # size of morph filter
    morph_filter = np.ones((filter_size,filter_size))
    bw = [cv2.dilate(cv2.erode(t_plane.astype('uint16'),morph_filter), morph_filter) for t_plane in bw]

    # Explicitly get rid of small stuff
    #bw = [morphology.remove_small_objects(t_plane.astype, min_size=25) for t_plane in bw]

    # Now add a buffer
    filter_size = 20 # size of morph filter
    morph_filter = np.ones((filter_size,filter_size))
    bw = [cv2.dilate(t_plane,morph_filter) for t_plane in bw]


    # Get bounding boxes
    bboxes = [measure.regionprops(t_plane.astype('int8')) for t_plane in bw]


    # Overlay bounding boxes on top of max int projections
    if do_plot:
        fig = plt.figure('bboxes1')
        fig.clf()
        axs = fig.subplots(1,2)
        maxValPlot = 7000

        [t_ax.imshow(t_plane, cmap='gray', vmax=maxValPlot) for t_plane, t_ax in zip(planes,axs)]
        [overlayBbox(t_ax,t_box) for t_box, t_ax in zip(bboxes,axs)]



        # Overlay bounding boxes on top of select stack planes
        fig = plt.figure('bboxes2')
        fig.clf()
        axs = fig.subplots(1,2)

        axs[0].imshow(im[100,:,:], cmap='gray', vmax=maxValPlot)
        axs[1].imshow(im[:,180,:], cmap='gray', vmax=maxValPlot)
        [overlayBbox(t_ax,t_box) for t_box, t_ax in zip(bboxes,axs)]


    return bboxes



def overlayBbox(ax,b) -> None:
    """
    Overlay a bounding box on an image (see get_crop_box)
    """

    minr, minc, maxr, maxc = b[0].bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '--r', linewidth=2.5);



def register_shutters(ds_stacks, metadata):
    """
    Register one shutter onto the other

    Inputs
    ds_stacks : the downsampled image stacks
    metadata : the meta-data of the full size stacks with which they are associated

    """


    print('Registering downsampled left and right images')
    #RES = sitk.Elastix(
    #        sitk.GetImageFromArray(ds_stacks[0]),
    #        sitk.GetImageFromArray(ds_stacks[1]),
    #        'translation')

    # The following allows us to more clearly set which is the fixed and moving image
    # It's now saving iteration files to disk, which is annoying.
    dir_name = 'lr_reg';
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    else:
        # Delete all files in it
        all_files = glob(os.path.join(dir_name,'*'))
        [os.remove(t_file) for t_file in all_files]


    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetOutputDirectory(dir_name)
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.LogToFileOn()
    elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(ds_stacks[0]))
    elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(ds_stacks[1]))
    elastixImageFilter.SetParameterMap(
        sitk.ReadParameterFile('translate_left_right_sheets_params.txt'))
    elastixImageFilter.Execute()

    # Read the saved parameter file
    p = sitk.ReadParameterFile(os.path.join(dir_name,'TransformParameters.0.txt'))

    # We will now change the parameters. TODO: we need a nice method to wrap this as it
    # is very clunky right now as the parameters are not a dict.
    #  https://github.com/SuperElastix/SimpleElastix/issues/169

    # Upscale parameters and zero the first, which must be zero. The sample can't
    # translate along the lightsheet axis.
    t_params = np.array(list(map(float, p['TransformParameters'])))
    t_params = t_params*4;
    t_params[0] = 0;
    p['TransformParameters'] = tuple(map(str,t_params)) # Convert to tuple of strings


    # Change the image size (TODO we will write new metadata for cropped)
    print('Loading full size stack to transform')
    fname = 'crop_' + metadata[0]['image_file_name']
    im = io.imread(fname)
    sh = im.shape
    sh = (sh[2],sh[1],sh[0])
    p['Size'] = tuple(map(str,sh)) # Convert to tuple of strings

    print('Transforming full size stack')
    RES = sitk.Transformix(sitk.GetImageFromArray(im),p)

    return sitk.GetArrayFromImage(RES)




def make_final_image(registered_stack, fixed_stack_metadata):
    """
    Create the final merged image. Flip so it's coronal planes.
    Make ds stack at 25 microns isotropic

    Inputs
    registered_stack : output of register_shutters
    fixed_stack_metadata : the meta-data for the image we registered *to*
    """

    print('Loading fixed image')
    im_f = io.imread('crop_' + fixed_stack_metadata['image_file_name'])

    print('Making blended image')
    blended = blend_along_axis(im_f,registered_stack,axis=2)
    blended = make_zaxis_first_axis(blended)

    # Save it
    print('Saving and creating downsampled stack')
    fname = re.sub('_Sh[01]','', fixed_stack_metadata['image_file_name'])

    fixed_stack_metadata['image_file_name'] = fname
    io.imsave('blended_' + fname, blended)
    create_downsampled_stack(fixed_stack_metadata,blended)

    # TODO -- remove the cropped images and place the originals in a sub-folder
    # TODO -- then eventually write the code such that the cropped images are never saved
    return blended





def make_zaxis_first_axis(data):
    """
    Flip dimensions and rotate

    data : image stack
    """
    data = np.transpose(data, (1,2,0))
    data = np.flipud(data)
    data = np.rot90(data, 3, (1,2))
    return data


def blend_along_axis(im1, im2, axis=0,
                     blending_center_coord=None,
                     pixel_width=None,
                     weight_at_pixel_width=0.95,
                     weight_threshold=1e-3,
                    ):

    """Sigmoidal blending of two arrays of equal shape along an axis

    Parameters
    ----------
    im1 : ndarray
        input array with values to keep at starting coordinates
    im2 : ndarray
        input array with values to keep at ending coordinates
    axis : int
        axis along which to blend the two input arrays
    blending_center_coord : float, optional
        coordinate representing the blending center.
        If None, the center along the axis is used.
    pixel_width : float, optional
        width of the blending function in pixels.
        If None, 5% of the array's extent along the axis is used.
    weight_at_pixel_width : float, optional
        weight value at distance `pixel_width` to `blending_center_coord`.
    weight_threshold : float, optional
        below this weight threshold the resulting array is just a copy of
        the input array with the highest weight. This is faster and more
        memory efficient.

    Returns
    -------
    blended_array : ndarray
        blended result of same shape as input arrays

    Notes
    -----
    This function could require less memory by blending plane by plane
    along the axis.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import ndimage
    >>> im = np.random.randint(0, 1000, [7, 9, 9])
    >>> im = ndimage.zoom(im, [21]*3, order=1)
    >>> degrad = np.exp(-0.01 * np.arange(len(im)))
    >>> im_l = (im.T * degrad).T.astype(np.uint16)
    >>> im_r = (im.T * degrad[::-1]).T.astype(np.uint16)
    >>> blend_along_axis(im_l, im_r, axis=0, pixel_width=5)

    Author - Marvin Albert
    Code taken from: https://gist.github.com/m-albert/bb4fa38436760c4e6d171239fb3e16c4
    """

    # use center coordinate along blending axis
    if blending_center_coord is None:
        blending_center_coord = (im1.shape[axis] - 1)/2

    # use 10% of extent along blending axis
    if pixel_width is None:
        pixel_width = im1.shape[axis]/20

    shape = im1.shape

    # define sigmoidal blending function
    a = -np.log((1 - weight_at_pixel_width) / weight_at_pixel_width) / pixel_width
    sigmoid = 1/(1 + np.exp(-a * (np.arange(shape[axis]) - blending_center_coord)))
    sigmoid = sigmoid.astype(np.float16)

    # swap array axes such that blending axis is last one
    im1 = np.swapaxes(im1, -1, axis)
    im2 = np.swapaxes(im2, -1, axis)

    # initialise output array
    out = np.zeros_like(im1)

    # define sub threshold regions
    mask1 = sigmoid < weight_threshold
    mask2 = sigmoid > (1 - weight_threshold)
    maskb = ~(mask1 ^ mask2)

    # copy input arrays in sub threshold regions
    out[..., mask1] = im1[..., mask1]
    out[..., mask2] = im2[..., mask2]

    # blend
    out[..., maskb] = (1 - sigmoid[maskb]) * im1[..., maskb] + \
                      sigmoid[maskb] * im2[..., maskb]

    # rearrange array
    out = np.swapaxes(out, -1, axis)

    return out




def create_downsampled_stack(metaDataStruct, stack=None):
    """


    This function runs the axis re-ording to get coronal
    sections, saves it, then makes a 25 micron downsampled stack
    in the same folder.

    Inputs
    metaDatStruct :
        The structure returned by mesospim_python_tools.io.return_mesoSPIM_files_in_path
    stack : Optionally the stack associated with metaDataStruct. None by default. If None,
        the data are loaded based on the meta-data
    do_isotropic : if true we make 25 micron stacks. If false we just downsample by taking
            every other pixel. This is for the registration.
    """

    fname = metaDataStruct["image_file_name"]

    if stack is None:
        stack = io.imread(fname)

    # Now we down-sample
    xy = metaDataStruct["meta_data"]["CFG"]["Pixelsize_in_um"]
    z = metaDataStruct["meta_data"]["POSITION"]["z_stepsize"]
    rescale_by =  np.array([xy, z, xy]) / 25
    ds = zoom(stack,rescale_by, order=1)


    ds_fname = os.path.join(dsRoot,dsSub,'ds_25micron_'+fname)

    print('Saving %s' % ds_fname)
    makeDownsampledDir()
    io.imsave(ds_fname,ds)


def makeDownsampledDir():
    """
    Make downsampled directory for 25 micron stacks in current directory
    """
    if not os.path.isdir(dsRoot):
        os.mkdir(dsRoot)

    if not os.path.isdir(os.path.join(dsRoot,dsSub)):
        os.mkdir(os.path.join(dsRoot,dsSub))

