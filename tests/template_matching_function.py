def match_corr(corr_obj, path):
    """
    return the center coordinates of the location of 'corr_obj' in 'img'.
    :param corr_obj: 2D numpy array of size [H_obj x W_obj] 
                     containing an image of a component.
    :param img: 2D numpy array of size [H_img x W_img] 
                where H_img >= H_obj and W_img>=W_obj, 
                containing an image with the 'corr_obj' component in it.
    :return:
        match_coord: the two center coordinates in 'img' 
                     of the 'corr_obj' component.
    """
    # ====== YOUR CODE: ======
    
    # Convert to float and normalize to [0, 1]
    normalized_img = img.astype(np.float32) / 255.0
    
    normalized_corr_obj = corr_obj.astype(np.float32) / 255.0
    normalized_corr_obj -= np.mean(normalized_corr_obj)
    # Apply filter with constant border padding (default value 0)
    output_image = cv2.filter2D(
        src = normalized_img,
        ddepth = -1,  # same depth as source
        kernel = normalized_corr_obj,
        borderType = cv2.BORDER_CONSTANT
        )
    #plt.imshow(output_image, cmap='gray')
    corr_obj_square_val = np.sum(normalized_corr_obj * normalized_corr_obj)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(np.abs(output_image-corr_obj_square_val))
    y,x = (min_loc[1], min_loc[0])  # (y, x)
    match_coord = (y,x)
    # ========================
    
    return match_coord