"""please put the module declarations here!"""
from enum import Enum
import os, sys, cv2, csv
import numpy as np



class Addresses(Enum):

  # please hecking change the next line to fit your system's local repo
  main_folder:        str = r'/Users/gianfajardo/Documents/Senior Design Project/Database of license plate images/License_Plate_Detection_Database'

  a_source_plates:    str = main_folder + r'/a_images'             	# ext: png/jpg
  b_cropped_plates:   str = main_folder + r'/b_cropped_plates'
  c_cropped_letters:  str = main_folder + r'/c_cropped_letters'			# ext: png/jpg
    

# checks if all the filepath addresses exist
for each_address in Addresses:

	if not os.path.exists(each_address.value):
		print(f'{each_address.name}\'s filepath does not exist. Please change this address.')
		sys.exit()

print(f'all folders found')



# use this 
# addrs: Addresses

letters = "ABCDEFGHIJKLMNPQRSTUVWXYZ"
numbers = "0123456789"
symbols = "-"

# california_LP_letters = letters + numbers
california_LP_letters = numbers + letters

image_dims = (108, 50)

num_regions = 6




# processing utilities -------------------------------------------------------------------------------------------
def find_letter_images(
    letters_folder:     str,
    letters_to_analyze: str,
		letter_extensions: 	tuple
  ) -> list:
  """
  Finds all of the images in a specified folder to return an iterable. useful for for-loops

  Args:
    letters_folder (str): address of the folder
    letters_to_analyze (str): filter variable; contains all of the characters I'd ever need

  Returns:
    list of all the addresses of each file
  """

  os.chdir(letters_folder)
  files = os.listdir(letters_folder)

  # filter method for excluding any non-image files and illegal CA characters
  img_path_list = []
  for each_file in files:
    file_path = os.path.join(letters_folder, each_file)


    # NOTE: each_file[0] is an easy way of checking the letter on Windows and MacOS.
    #       If any copies made on Windows are made, please make sure the version number
    #       is after the type of letter or number.
    if each_file.endswith(letter_extensions)  and  each_file[0] in letters_to_analyze:
      img_path_list.append((file_path, each_file))

  # img_path_list.sort()
  return img_path_list


def find_images(
    letters_folder:     str,
		letter_extensions: 	tuple
  ) -> list[tuple[str, str]]:
  """
  Finds all of the images in a specified folder to return an iterable. useful for for-loops

  Args:
    letters_folder (str): address of the folder

  Returns:
    list of all the addresses and names of each file
  """

  os.chdir(letters_folder)
  files = os.listdir(letters_folder)

  # filter method for excluding any non-image files and illegal CA characters
  img_path_list: list[tuple[str, str]] = []
  for each_file in files:
    file_path = os.path.join(letters_folder, each_file)
    
    if each_file.endswith(letter_extensions):
      img_path_list.append((file_path, each_file))

  return img_path_list


class SubRegionCalculator:
	"""
	TODO: add another class as a constructor input which describes a set of subdivisions of the image
	"""

	def __init__(self):
		pass

	def calc_subregion_densities(
			self,
			image_cv2: np.ndarray
		) -> np.array:
		"""This is a description of the function.\n
		Args:
				image_cv2 (numpy.ndarray): arraylike format in the grayscale format
				
		Returns:
				density_array (np.array): returns the density of black pixels of each subregion.
		"""

		"""pre-definitions"""
		zeros_collection  = 	np.zeros(image_cv2.shape, dtype=np.uint8)
		ones_collection   = 	np.ones( image_cv2.shape, dtype=np.uint8)
		canvas           	=  np.matrix(image_cv2.shape, dtype=np.uint8)

		image = np.asarray(image_cv2, dtype=np.uint8)
		# print(image)

		
		# TODO: perhaps modify this code so it's using proper masked arrays?
		# link: https://numpy.org/doc/stable/reference/routines.ma.html
		# checks if all entries in the img array at a certain pixel is white
		bg_mask     = (image == 255)
		text_mask   = np.logical_not(bg_mask) # returns logical array
		# print(black_mask)
		

		canvas = np.add(
				zeros_collection,
				text_mask * ones_collection
		)

		img_height, img_width = image.shape[:2]
		img_height  = np.uint16(img_height)
		img_width   = np.uint16(img_width)
		

		# this np.arange() function is declaring equally sized subsections of the width and height
		# TODO: allow for declaring non-equally sized subregions. specify this information in the constructor
		# TODO: declare separate width and height parameters instead of using only "num_regions" throughout
		subdivision_list_w = np.arange(0, num_regions+1).reshape(1, num_regions+1)
		subdivision_list_h = np.column_stack(subdivision_list_w)
		
		width_list  = np.uint16(subdivision_list_w *  img_width/num_regions)
		height_list = np.uint16(subdivision_list_h * img_height/num_regions)
		# print(width_list)
		# print(height_list)



		"""This calculates the lower and upper bounds"""
		# row axis
		upper_bound_list_h = height_list
		lower_bound_list_h = height_list + np.ones(upper_bound_list_h.shape, dtype=np.uint8)
		
		lower_bound_list_h = lower_bound_list_h[0:num_regions  , :]
		upper_bound_list_h = upper_bound_list_h[1:num_regions+1, :]

		# column axis
		lower_bound_list_w = width_list + np.ones(width_list.shape, dtype=np.uint8)
		upper_bound_list_w = width_list

		lower_bound_list_w = lower_bound_list_w[:, 0:num_regions  ]
		upper_bound_list_w = upper_bound_list_w[:, 1:num_regions+1]



		"""This calculates the area of black pixels for each subsection"""
		density_array = np.zeros((num_regions, num_regions))

		for j in range(num_regions):        # col
			for i in range(num_regions):    	# row
					
				# np.intp() is a datatype that's preferred for indexing apparently
				dims = np.array(
						[ 
              lower_bound_list_h[i,0],  # lower_h / dims[0]
							upper_bound_list_h[i,0],  # upper_h / dims[1]

							lower_bound_list_w[0,j],  # lower_w / dims[2]
							upper_bound_list_w[0,j]   # upper_w / dims[3]
							
						], dtype=np.intp
				)

				area_of_analysis = canvas[ dims[0]:dims[1], dims[2]:dims[3] ]

				# print(f'from x: ({dims[0]}, {dims[1]}) \t& \ty: ({dims[2]}, {dims[3]})')
				# print(area_of_analysis, '\n')

				density_array[i,j] = np.sum(area_of_analysis)
		


		"""This calculates the area of each subsection"""
		# this calculates the dimensions of each side
		width_dimensions_list = (  width_list[:, 1:num_regions+1]
														 - width_list[:, 0:num_regions  ])

		# creates a 6-length array 
		height_dimensions_list = (  height_list[1:num_regions+1, :]
															- height_list[0:num_regions  , :])

		# print(height_dimensions_list)
		# print(width_dimensions_list)

		
		# this will be our denominator
		area_array = np.matmul(height_dimensions_list, width_dimensions_list)
		# print(area_array)

		density_array = np.divide(density_array, area_array)

		return density_array






def read_column_by_header(
		filename: 		str,
		column_name:	str
	) -> list:
  """Reads a specific column from a CSV file based on its header.

  Args:
    filename (str): The name of the CSV file.
    column_name (str): The name of the column to read.

  Returns:
    list: A list containing the values from the specified column.
  """

  with open(filename, 'r') as csv_file:
    reader = csv.DictReader(csv_file)

    column_data = []
    for row in reader:
      column_data.append(row[column_name])
    return column_data


def find_foreground_mask(
		file_path: str
	) -> np.array:
	"""This is a description of the function.\n
	Args:
		file_path (str): local file path of the image that's being analyzed.
			
	Returns:
		foreground_coords_list (np.array): returns the density of black pixels of each subregion.
	"""
	
	# Read the image
	image_cv2 = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
	foreground_coords_list = np.argwhere(image_cv2 == 0)

	return foreground_coords_list



  
def within_bounds(height: int, width: int) -> bool:
  # these are the parameters that we can use to tune this function
  # minimum ratios that the program has to recognize
  # aspect ratio: h/w
  min_aspect_ratio, max_aspect_ratio = 1.5, 3.0
  max_width       = 66
  max_height = max_aspect_ratio*max_width
  min_width      = 30
  min_height = min_aspect_ratio*min_width

  return      (min_aspect_ratio < float(height/width) < max_aspect_ratio) \
          and (min_height < height < max_height) \
          and (min_width < width < max_width)


def segment_characters(image: tuple) -> None:
  """Segment characters from a license plate.

  Args:
    image_path (tuple): Path to the license plate image.

  Returns:
    list: List of segmented character images.
  """

  letter_extensions = ('.png','.jpg')

  image_path, image_name = image

  print(image_name)

  # load the image
  img = cv2.imread(image_path, 0)
  plate_height, plate_width = img.shape


  # preprocessing using the group's functions
  # img = process.adjust_brightness(img)
  # img = process.grayscale(img)
  # img = process.unsharp_mask(img, sigma=0.5, strength=2.0)


  _, thresholded_img = cv2.threshold(
      img,
      200,
      255,
      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
  )


  kernel = np.ones((4,4), np.uint8)
  thresholded_img = cv2.erode(thresholded_img, kernel, iterations=1)  # Erode to break connections
  thresholded_img = cv2.dilate(thresholded_img, kernel, iterations=1)  # Dilate to restore character size

	# corrects the image if it looks wrong; otherwise do not do anything
  # img_corrected_bg = invert_if_black_background(thresholded_img)
  img_corrected_bg = thresholded_img.copy()
  
  
  # Find contours
  contours, _ = cv2.findContours(thresholded_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
  thresh_with_color = cv2.cvtColor(img_corrected_bg.copy(), code=cv2.COLOR_GRAY2BGR)

  # print(contours)

  # sys.exit()

  img_with_boxes = thresh_with_color.copy()
  

  # Filter contours based on hierarchy and size
  char_regions = []
  # char_regions: list[tuple[int, int, int, int]]
  for each_contour in contours:
    x, y, w, h = cv2.boundingRect(each_contour)
    # print(f'(w, h) = ({w}, {h}), h/w = {float(h/w)}')

    if within_bounds(height=h, width=w):
      # print(f'passed filter 1')

      if (h < plate_height/1.1) and (w < plate_width/2):  # Adjust minimum size as needed
        char_regions.append((x, y, w, h))

        img_with_boxes = cv2.rectangle(
            thresh_with_color,
            (x, y),
            (x+w, y+h),
            (0, 0, 255),
            4
        )
        # print('passed filter 2; bounding boxes drawn')

        # print(f'\t(w, h) = ({w}, {h}), h/w = {float(h/w)}')


  cv2.imshow('Image with Bounding Boxes', img_with_boxes)
  cv2.waitKey(300)
  

  prompt: str = ""
  while len(prompt) != 1:
    prompt = input(f'invert (i/I), or non-invert (n/N), or skip (?)? ')

    if prompt in 'iI':
      final_img = cv2.bitwise_not(img_corrected_bg)
    elif prompt in 'nN':
      final_img = img_corrected_bg
      
    elif prompt in '?':
      return

    else: 
      print('Please enter a valid input')
    
  cv2.imshow('Final Image', final_img)
  

  for each_letter_extension in letter_extensions:
    image_name.replace(each_letter_extension, '')


  # sorts the characters from left to right
  sorted_char_regions = sorted(char_regions, key=lambda x: x[0])

  

  n: int = 0
  for x, y, w, h in sorted_char_regions:
    roi = final_img[y:y+h, x:x+w]
    resized_output = cv2.resize(roi, (50, 108))
    
    cv2.imshow('Characters', roi)
    cv2.waitKey(10)


    # asks for the character
    output_name: str = ""
    while len(output_name) != 1:
      output_name = input(f'letter? ').upper()

    if output_name in '?':
      continue
    else:
      n += 1


    cv2.imwrite(f'{output_name} - {image_name}.png', resized_output)
    cv2.waitKey(100)

    # sys.exit()
  

  if n == 7:
    print('\tsuccess\n\n')
  else:
    print('\tfailed\n\n')
    



    

		
def export_data(
		data_folder:	str,
		data_fields:	list,
		data_rows:  	str
	) -> None:

	'''
	Function which exports our data in a csv format to a specified folder.\n
	Args:
		data_folder (str): file path of the folder
		data_fields (list): lists of strings
		data_rows (str): 
	Returns:
		None
	'''

	os.chdir(data_folder)
	all_files = os.listdir(data_folder)


	# finds the latest session and creates a name for the new session
	n 			= int()
	last_n 	= int(1)
	for each_file in all_files:

		# filters other csv files that this function did not create
		if each_file.startswith('session_') and each_file.endswith('.csv'):

			# extracts the number from the image name
			n = int( each_file[ 8:(len(each_file) - 4) ] )

			if n > last_n:
				last_n = n
	

	# get the response from the user on which file to overwrite
  # TODO: disable temporarily
	# user_input = input(f'Overwrite latest session_{last_n}.csv? ')
	# if user_input in 'xX':
	# 	sys.exit()
	# elif user_input in 'nN':
	# 	last_n = last_n+1
	# elif user_input in 'yY':
	# 	pass 

	
	# writing to csv file with new_filename
	new_filename = f'session_{last_n}.csv'
	with open(new_filename, 'w') as csv_file:

		# creating a csv writer object
		csvwriter = csv.writer(csv_file)

		# writing the fields and all rows
		csvwriter.writerow(data_fields)
		csvwriter.writerows(data_rows)
          







def read_csv_file(
    address:            str,
    filename:           str,
    selected_features:  list[str]
    
) -> tuple[list[list[float]], list[str]]:
  """
  reads the csv file which is expected to have the dataset with features
  
  Args:
      address (str): the address of the csv file
      filename (str): the name of the csv file
      selected_features (list[str]): a list of targets to include
  
  Returns:
      features ( list[list[float]] ): the binary features which are linked to the target
      targets ( list[str] ): the expected target output of the dataset
  """
  
  features: list[list[float]] = []
  targets:  list[str]         = []
  
  os.chdir(address)
  with open(filename, 'r') as csv_file:

    reader = list(csv.reader(csv_file))
    header = reader[0]        # header of the .csv file
    data_rows = reader[1:-1]  # Skip header and last row

    # get indices of selected features (e.g., RS=1, LS=2, TS=4)
    feature_indices = [header.index(feat) for feat in selected_features]

    for each_row in data_rows:
      target = each_row[0]  # Extract target (first column)

      # extract ONLY the selected features
      feature_vals = [float(each_row[i]) for i in feature_indices]
      
      features.append(feature_vals)
      targets.append(target)
  
  # for each_feature_results in features:
  #   print(each_feature_results)
  # sys.exit()
  
  return (features, targets)