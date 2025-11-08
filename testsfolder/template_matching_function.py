import numpy as np
import cv2

def fix_size(image, template_width):
    """
    Resizes image to target width while preserving aspect ratio.
    
    :param image: Input image array
    :param template_width: Target width
    :return: Resized image
    """
    h, w = image.shape[:2]
    aspect_ratio = h / w
    target_width = template_width
    target_height = int(target_width * aspect_ratio)

    # Resize while preserving aspect ratio
    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    return resized_image

KERNEL_SIZE=31
def pre_proccess(image, template_width):
    """
    Preprocesses image by resizing, normalizing, and applying Gaussian blur.
    
    :param image: Input image array
    :param template_width: Target width to resize to (preserving aspect ratio)
    :return: Preprocessed image
    """
    image_width = image.shape[1]
    if(image_width != template_width):
        image = fix_size(image, template_width)
    image = image.astype(np.float32) / 255.0
    image -= image.mean(axis=(0,1))
    if(np.std(image) != 0):
        image = image / np.std(image)
    image = cv2.GaussianBlur(image, (KERNEL_SIZE, KERNEL_SIZE), 0)  # kernel size (5,5), sigma=0
    # image = cv2.equalizeHist(image)
    
    return image

def perform_correlation(template_obj, image):
    """
    Performs correlation with horizontal sliding by padding the larger image.
    
    Note: Both images have the same width after resize. To allow horizontal sliding,
    we add padding to enable template matching at different horizontal offsets.
    """
    # Calculate intersection area (width is same, height is min)
    template_h, template_w = template_obj.shape[:2]
    image_h, image_w = image.shape[:2]
    
    intersection_h = min(template_h, image_h)
    intersection_area = intersection_h * template_w  # width is same for both
    
    # Calculate bigger image area
    bigger_area = max(template_h * template_w, image_h * image_w)
    
    # Determine which is smaller (template) and larger (image) for padding
    # Crop both to intersection height first
    template_cropped = template_obj[:intersection_h, :]
    image_cropped = image[:intersection_h, :]
    
    # Add horizontal padding to allow sliding (pad the larger image to enable template sliding)
    # Padding amount: half of template width on each side
    pad_x = template_w // 2
    image_padded = cv2.copyMakeBorder(
        image_cropped,
        top=0, bottom=0,
        left=pad_x, right=pad_x,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    
    #calc the sum of the squared pixels for each color separately
    corr_obj_square_val = np.sum(template_cropped ** 2, axis=(0, 1))

    output_image = np.zeros((intersection_h, image_padded.shape[1], 3), dtype=np.float32)
    # Apply filter with the padded image to allow horizontal sliding
    for c in range(3):  # for R, G, B channels
        output_image[:, :, c] = cv2.filter2D(
            src=image_padded[:, :, c],
            ddepth=-1,  # same depth as source
            kernel=template_cropped[:, :, c],
            borderType=cv2.BORDER_CONSTANT
        )
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(np.mean(np.abs(output_image - corr_obj_square_val), axis=2))
    
    # Apply penalty: bigger_image_area / intersection_area
    penalty = bigger_area / intersection_area
    
    return min_val * penalty

def rotate_template(image, degrees):
    """
    Rotates image by specified degrees around its center.
    
    :param image: Input image array
    :param degrees: Rotation angle in degrees
    :return: Rotated image
    """
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
        normalized score between 0 and 1 (lower is better match)
    """
    # Determine max width and resize smaller image to match larger
    template_w = template_obj.shape[1]
    image_w = image.shape[1]
    target_width = max(template_w, image_w)
    
    # Convert to float and normalize to larger width
    template_obj = pre_proccess(template_obj, target_width)
    image = pre_proccess(image, target_width)

    deg_arr = [0,3,-3]
    min_vals_rot = np.zeros(len(deg_arr))
    for i in range(len(deg_arr)):
        degrees = deg_arr[i]
        rotated_template = rotate_template(template_obj, degrees)
        min_vals_rot[i] = perform_correlation(rotated_template, image)
    
    #print(min_vals_rot)
    min_val = min_vals_rot.min()
    
    # Normalize to [0, 1] range with clamping
    MAX_EXPECTED_SCORE = 10000.0  # Fixed range based on expected correlation values
    normalized = min(1.0, min_val / MAX_EXPECTED_SCORE)
    
    return normalized


# template = cv2.imread("photos/person_id_13_capture_1.png", cv2.COLOR_BGR2RGB)
# template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

# min_vals = np.zeros(20)
# for i in range(19):
#     image = cv2.imread(f"photos/person_id_13_capture_{i+1}.png", cv2.COLOR_BGR2RGB)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     min_vals[i] =  template_match(template, image)

# image = cv2.imread(f"photos/person_id_59_capture_11.jpg", cv2.COLOR_BGR2RGB)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# min_vals[19] =  template_match(template, image)

# min_val_index = np.argmin(min_vals)
# np.set_printoptions(precision=2, suppress=True)
# print(f"min score: {min_vals[min_val_index]}, photo: person_id_13_capture_{min_val_index+1}")
# print(f"all scores: {np.round(min_vals, 2)}")
# print(f"ratio bad/good is: {min_vals[19] / (min_vals[1:19].min())}")