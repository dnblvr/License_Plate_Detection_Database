# License_Plate_Detection_Database

Author: Gian Fajardo

This is the [database](https://universe.roboflow.com/amykun-qoz6t/license-plate-recognition-8fvub) by Amykun from Roboflow that the machine learning subgroup (Bryan, Leo, and I) used for test characters. Keep in mind that this functioned as a license plate detection dataset. Because we needed characters flat against the screen, these characters had to be extracted manually using a procedure, which was assigned to Leo and I.

There were three broad categories of image for license plate detection purposes. We included and excluded plates and characters depending on the following three categories.

## International License Plate Images

There were the international-based license plates. These were excluded from the process entirely.

![International plate whose ID says 'Roma EO5426'](a_invalid_images/0_6_hr_png_jpg.rf.cc7ebcb67b73d53500e915d148e2677d.jpg)
![International plate whose ID says 'WAE92D'](a_invalid_images/0_15_hr_png_jpg.rf.3f683f751985139c09e087e25dff7ba6.jpg)
## US License Plate Images

There were license plates from California. There were others from other states. Because we just needed the dataset, these types of plate were processed, and some were discarded depending on if they were similar enough to the CA styling. Here are some examples:

!['GDKARMA'](a_images/0a4ada79-be27-4c1e-8243-31dc9c35c134_jpg.rf.9d0c46f14dbea14bc1c4c1bbcb91a5f4.jpg)
!['LVN SLO'](a_images/0af39d30-64c3-44f7-a5de-2ea676f02381_jpg.rf.8e2144f61977535ef5efce2c775f952a.jpg)
![''ROCKESQ](a_images/0ba0aa78-a3f8-43a3-9b72-0f47b34ecc94_jpg.rf.82c40982db0b2ba7a0209a797cb3bd07.jpg)
![Texan plate that says 'HVT1972'](a_images/1_jpg.rf.78e1ed7a02f2c2ca5af34ab9b5bc61de.jpg)
!['ACTS2:38'](a_images/130_0056_jpg.rf.6c53f0a37fea46f0fbf6a1fc7aa0d459.jpg)


## Multi-License Plate Images

Still, there were some test images like these. These were still included in the batch if they had a CA license plate within it. For example, there are these two images:

![multi-plate image, two CA plates reading 'LVTOPLY' and 'GBAYPKR'](a_images/0a2bdc41-3f28-4ea8-b30f-74eebbe8e7ae_jpg.rf.8664b22e78cc3af62a0d2bd1e5d28f4c.jpg)
![multi-plate image, two CA plates reading 'WNTGVUP' and 'GRLHIKR'](a_images/0c66200c-90bc-49a8-a560-db684298057b_jpg.rf.bafc27cd14c9b4c546a3a81fcfaa0f72.jpg)

## The Process

Run `perspective_transform.py`. You will be instructed to find the four corners of the license plate in the image. Start with clicking on the matplotlib figure in the specified order:

1. the top-left corner, then
2. the top right corner, then
3. the bottom-right corner, and finally
4. the bottom-left corner.

You will receive feedback when the terminal gives the coordinates of the figure.

![alt text](screenshots/image.png)

This realigns the plates and places the folder onto `<main_folder>/b_cropped_plates` folder.

---

Next, run `segmentation.py`. You will be given three windows, each of which will contain: the non-inverted image, the resulting images, and the character that the segmentation has given. You will be instructed to invert the image, leave as is, or skip the image. Type onto the terminal the choice you want.

Once the choice is picked, the resulting image is displayed. Make sure that the foreground is black and the background is white. if the first two options are picked, give the letter information onto the terminal. Once done, they should go to `<main_folder>/c_cropped_letters` folder.
