import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import header as utils


def main():

  class Click_Finder():
    """
    this class exists so that I can record the following attribute:
      num_clicks"""
    num_clicks = 0

    # function to display coordinates on hover
    def on_mouse_click(self, event):

      # Condition 1: if the user clicks within the graph
      if event.inaxes:
        x, y = event.xdata, event.ydata

        print(f"\tPixel coordinates: ({int(x)}, {int(y)})")
        corner_coords[self.num_clicks, 0] = x
        corner_coords[self.num_clicks, 1] = y

        # with this line, it always increments and resets!
        self.num_clicks = (self.num_clicks + 1) % 4

        if self.num_clicks == 0:
          plt.disconnect(cid)
          plt.close()
      

      # Condition 1: if the user clicks outside the graph
      else:
        print(f"this image is skipped")

        plt.disconnect(cid)
        plt.close()

  
  cf = Click_Finder()

  gap_len = 40

  # desired height and width of the plate
  plate_height, plate_width = 6+3/8, 12+1/4
  height, width = int(plate_height*40), int(plate_width*40)


  files_list = utils.find_images(
      letters_folder      = utils.Addresses.a_source_plates.value,
		  letter_extensions   = ('.png', '.jpg')
  )


  # do not remove as this will write to the source folder
  os.chdir(utils.Addresses.b_cropped_plates.value)

  k: int = -1
  for each_img, each_filename in files_list:

    k = k+1

    destination_filepath = utils.Addresses.b_cropped_plates.value + r'/' + each_filename

    # print(f"{destination_filepath}")

    # if the file is copied over, then no need to remake it
    if os.path.exists(destination_filepath):
      print(f"skip\t\timg {k}")
      continue
    else:
      print(f"continue\timg {k}")

    # sys.exit()

    # read the image
    img     = cv2.imread(each_img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # makes the white border around the new image
    bordered_image = cv2.copyMakeBorder(
        src         = img_rgb,
        top         = gap_len,
        bottom      = gap_len,
        left        = gap_len,
        right       = gap_len,
        borderType  = cv2.BORDER_CONSTANT,
        value       = [255,255,255]
    )

    # print(each_filename)

    # display the image
    fig	= plt.figure() 
    plt.imshow(bordered_image)

    # store the original coordinates of the plate
    corner_coords = np.empty((4,2), dtype=np.float32)

    # connect the mouse motion event to the function
    cid = fig.canvas.mpl_connect('button_press_event', cf.on_mouse_click)

    plt.show()


    # define the original & desired perspective points
    lp_coords = np.float32(                # Desired  points
        [[0,     0],
        [width,  0],
        [width,  height],
        [0,      height]]
    )


    # get the perspective transform matrix
    pt_matrix   = cv2.getPerspectiveTransform(corner_coords, lp_coords)
    warped_img  = cv2.warpPerspective(
        src   = bordered_image,
        M     = pt_matrix,
        dsize = (width, height)
    )


    # display the warped image
    cv2.imshow("Warped Image", warped_img)
    cv2.imwrite(each_filename, warped_img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


main()