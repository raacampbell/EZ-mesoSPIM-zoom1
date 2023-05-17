# EZ-mesoSPIM-zoom1


The pipeline handles low-zoom mesoSPIM data acquired over multiple channels with left and right sheet separately.
It registers one sheet image onto the other and then produces a blended image. 





### One
* Crop the full size images by generating downsampled stacks that are just reduced by a factor of 4. 
* Finding crop boxes. 
* Do this all in RAM. 
* Save the cropped images.
```
import mesospim_python_tools
from ez_mesospim_zoom1 import ez_pipe

# Should be in sample directory before proceeding

# First we get the downsampled stacks for the background channel
# You must supply the correct laser line name for your background
ds_stacks = ez_pipe.downsample_channel('647 nm')

## Now we find the crop box (you may need to tweak parameters)
bboxes=ez_pipe.get_crop_box(ds647,otsu_scale=0.09, plot_max_val=500);

# Crop all stacks, making new cropped files
ds_stacks = ez_pipe.crop_full_stacks(bboxes)

Loading ECiBrain_LRtest_rightsideskew_Mag1x_Tile0_Ch488_Sh0_Rot0.tiff
Saving to crop_ECiBrain_LRtest_rightsideskew_Mag1x_Tile0_Ch488_Sh0_Rot0.tiff
Loading ECiBrain_LRtest_rightsideskew_Mag1x_Tile0_Ch488_Sh1_Rot0.tiff
Saving to crop_ECiBrain_LRtest_rightsideskew_Mag1x_Tile0_Ch488_Sh1_Rot0.tiff

```


### Two
Note: if the sheets are well registered you can skip this step
* Use the downsampled images from above to register one sheet to the other. 
* Use the parameters to transform full size stack. 
* Will have to double all numbers. 

```
#re-load the downsampled stacks from the cropped data
ds647c = ez_pipe.downsample_channel('647 nm',load_cropped=True)
transform_params = ez_pipe.register_shutters(ds647c)


ez_pipe.apply_transformation(tranform_params)
```

### Three
Blend and save blended image with coronal planes.
Make 25 micron stack from the blended image. 
```
FINAL = ez_pipe.make_final_image(RES,stacks[1])
```

### Enjoy
```
import napari
v = napari.view_image(FINAL)
```

Once we have an elegant thing that does this, we need to generalise to multiple channels
where the use defines which will be the background channel. So the above pipeline works
on that then the other channels are passed through the already calculated params. 




