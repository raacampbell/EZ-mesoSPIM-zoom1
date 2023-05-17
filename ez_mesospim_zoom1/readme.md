# What is it?

To assess whether we can register one sheet to the other following purposeful l/r misalignment.


In the misaligned dir we show that a translation cor with Elastix works well so 
long as we set the X param (first one) to zero. 

To do the full res reg we copy the parameters and change the size to new image size. e.g.
(Size 1710 738 1580)
This will use about 30 GB of a RAM for a brain sampled at 10 microns.

Seems to work but reg was on isotropic and data are not so re-jig nascent pipeline:

### Preliminary pipeline

Then we make downsampled stacks in current directory that are not isotropic or flipped. 
This is just so we can crop then register the full size stacks to each other. 

```
ez_pipe.runOnAllTIFFs(isotropic=False,reorg=False)
```


This makes:
```
FOR_REG_ECiBrain_LRtest_rightsideskew_Mag1x_Tile0_Ch488_Sh0_Rot0.tiff
FOR_REG_ECiBrain_LRtest_rightsideskew_Mag1x_Tile0_Ch488_Sh1_Rot0.tiff
```

We can get the crop boxes:
```
fname = ['FOR_REG_ECiBrain_LRtest_rightsideskew_Mag1x_Tile0_Ch488_Sh0_Rot0.tiff', 
         'FOR_REG_ECiBrain_LRtest_rightsideskew_Mag1x_Tile0_Ch488_Sh1_Rot0.tiff']

bb=ez_pipe.getCropBox(fname,do_plot=True)
```

Use the crop boxes to crop the full size stacks and also re-make the cropped images.
We have to use cropped images made from these stacks to do the registration.
```
s=mesospim_python_tools.io.return_mesoSPIM_files_in_path(os.getcwd())
ez_pipe.crop_full_stacks(s,bb)
```

Then we register:
```
$ mkdir sheet_reg
$ elastix -m FOR_REG_ECiBrain_LRtest_rightsideskew_Mag1x_Tile0_Ch488_Sh0_Rot0.tiff -f FOR_REG_ECiBrain_LRtest_rightsideskew_Mag1x_Tile0_Ch488_Sh1_Rot0.tiff -p ../elastix_params/translate_left_right_sheets_params.txt  -out sheet_reg/

```

We verify the sizes of the stacks:
```
im=io.imread('crop_ECiBrain_LRtest_rightsideskew_Mag1x_Tile0_Ch488_Sh0_Rot0.tiff')
im.shape
 (748, 1730, 1603)
```




Set the translation along sheet axis (first parameter) to zero and replace the image size with above. 
Editing a copy of the parameters placed in a new directory

```
mkdir sheet_reg
cp sheet_reg/TransformParameters.0.txt full_res/
```
 
So in this case the modified lines are:
```
//(Size 529 571 247)
(Size 1730 1603 748)
```

The translation needs to be multiplied by the amount we have downscaled it by. so:
```
(TransformParameters 1.262779 -0.529112 10.338310)
```

Means:
```
In [149]: np.array([1.262779, -0.529112, 10.33831]) / 0.33
Out[149]: array([ 3.82660303, -1.6033697 , 31.32821212])
```

And we replace:
```
//(TransformParameters 1.262779 -0.529112 10.338310)
(TransformParameters 0 -1.6 31.3)
```


```
transformix -in crop_ECiBrain_LRtest_rightsideskew_Mag1x_Tile0_Ch488_Sh0_Rot0.tiff -tp full_res/TransformParameters.0.txt -out full_res/
```

This yields nicely registered full size stacks. 

There is, however, a weirdness in the shapes. The transformed image of shape (748, 1603, 1730)
but the original is (748, 1730, 1603). Indeed the transformed image is shorter along image rows and ends at 
1603. However, it doesn't extend to 1730 along rows. They both seem to end at 1603.
1604. For now we hack it:
```
blended = blender.blend_along_axis(im_1[:, 0:1603, :], im_0[:, :, 0:1603], axis=2)
```

## 18th April
However it does work. So let's come up with a nicer pipeline.
`/dev_images` contains stacks which have been downsampled from 2048 to 1024, making the 
images far smaller. Still 10 micron steps. So we can run the whole pipeline very quickly. 
In the following we will be implicitly treating shutter 0 as the moving image throughout. 
The steps we want to implement are:


### One
* Crop the full size images by generating downsampled stacks that are just reduced by a factor of 2. 
* Finding crop boxes. 
* Do this all in RAM. 
* Save the cropped images.
```
import mesospim_python_tools
import ez_pipe
stacks=mesospim_python_tools.io.return_mesoSPIM_files_in_path(os.getcwd())
ds_stacks = ez_pipe.crop_full_stacks(stacks,do_plot=True)
Loading ECiBrain_LRtest_rightsideskew_Mag1x_Tile0_Ch488_Sh0_Rot0.tiff
Loading ECiBrain_LRtest_rightsideskew_Mag1x_Tile0_Ch488_Sh1_Rot0.tiff

Saving to crop_ECiBrain_LRtest_rightsideskew_Mag1x_Tile0_Ch488_Sh0_Rot0.tiff
Saving to crop_ECiBrain_LRtest_rightsideskew_Mag1x_Tile0_Ch488_Sh1_Rot0.tiff
```


### Two
* Use the downsampled images from above to register one sheet to the other. 
* Use the parameters to transform full size stack. 
* Will have to double all numbers. 

```
RES = ez_pipe.register_shutters(ds_stacks,stacks)
```

### Three
Blend and save blended image with coronal planes.
Make 25 micron stack from the blended image. 
```
FINAL = ez_pipe.make_final_image(RES,stacks[1])
```



Once we have an elegant thing that does this, we need to generalise to multiple channels
where the use defines which will be the background channel. So the above pipeline works
on that then the other channels are passed through the already calculated params. 




