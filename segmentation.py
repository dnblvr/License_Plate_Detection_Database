import os
import header as utils

def main():

  # print(f'{cropped_images_folder=}')

  # if not os.path.exists(utils.trained_characters):
  #   os.makedirs(utils.trained_characters)

  os.chdir(utils.Addresses.c_cropped_letters.value)
  
  # this is applied to flattened trained images
  image_paths = utils.find_images(
      letters_folder    = utils.Addresses.b_cropped_plates.value,
      letter_extensions = ('.png','.jpg')
  )

  os.chdir(utils.Addresses.c_cropped_letters.value)

  for image_path in image_paths:
    utils.segment_characters(image_path)

main()
  