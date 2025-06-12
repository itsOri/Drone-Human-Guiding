import numpy as np
import cv2

def fix_size(image, template_width):
    h, w = image.shape[:2]
    aspect_ratio = h / w
    target_width = template_width
    target_height = int(target_width * aspect_ratio)

    # Resize while preserving aspect ratio
    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    return resized_image

KERNEL_SIZE=15
def pre_proccess(image, template_width):

    image_width = image.shape[1]
    if(image_width != template_width):
        image = fix_size(image, template_width)
    image = image.astype(np.float32) / 255.0
    image -= image.mean(axis=(0,1))
    if(np.std(image) != 0):
        image = image / np.std(image)
    image = cv2.GaussianBlur(image, (KERNEL_SIZE, KERNEL_SIZE), 0)  # kernel size (5,5), sigma=0

    return image

def perform_correlation(template_obj, image):
    #calc the sum of the squared pixels for each color separately
    corr_obj_square_val = np.sum(template_obj ** 2, axis=(0, 1))

    output_image = np.zeros_like(image)
    # Apply filter with constant border padding (default value 0)
    for c in range(3):  # for R, G, B channels
        output_image[:, :, c] = cv2.filter2D(
            src=image[:, :, c],
            ddepth=-1,  # same depth as source
            kernel=template_obj[:, :, c],
            borderType=cv2.BORDER_CONSTANT
        )
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(np.mean(np.abs(output_image- corr_obj_square_val), axis=2))
    
    return min_val

def rotate_template(image, degrees):
    # Get image dimensions
    height, width = image.shape[:2]

    # Define center, angle, and scale
    center = (width // 2, height // 2)
    angle = degrees  # degrees
    scale = 1.0
        
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Apply rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderValue=(0, 0, 0))

    return rotated_image

def template_match(template_obj, image):
    """
    returns the best score of correlation between the template object and the image
    
    :param template_obj: 2D numpy array of size [H_obj x W_obj] 
                     containing an image of a component.

    :param image: 2D numpy array of size [H_img x W_img] 
                where H_img >= H_obj and W_img>=W_obj, 
                containing an image with the 'corr_obj' component in it.
    :return:
        match_coord: the two center coordinates in 'img' 
                     of the 'corr_obj' component.
    """
    # Convert to float and normalize to [0, 1]
    template_obj = pre_proccess(template_obj, template_obj.shape[1])
    image = pre_proccess(image, template_obj.shape[1])

    deg_arr = [0,1,-1,2,2,3,-3]
    min_vals_rot = np.zeros(len(deg_arr))
    for i in range(len(deg_arr)):
        degrees = deg_arr[i]
        rotated_template = rotate_template(template_obj, degrees)
        min_vals_rot[i] = perform_correlation(rotated_template, image)
    
    #print(min_vals_rot)
    min_val = min_vals_rot.min()
    return min_val


template = cv2.imread("photos/person_id_13_capture_1.png", cv2.COLOR_BGR2RGB)
template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

min_vals = np.zeros(20)
for i in range(19):
    image = cv2.imread(f"photos/person_id_13_capture_{i+1}.png", cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    min_vals[i] =  template_match(template, image)

image = cv2.imread(f"photos/person_id_59_capture_11.jpg", cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
min_vals[19] =  template_match(template, image)

min_val_index = np.argmin(min_vals)
np.set_printoptions(precision=2, suppress=True)
print(f"min score: {min_vals[min_val_index]}, photo: person_id_13_capture_{min_val_index+1}")
print(f"all scores: {np.round(min_vals, 2)}")
print(f"ratio bad/good is: {min_vals[19] / (min_vals[1:19].min())}")