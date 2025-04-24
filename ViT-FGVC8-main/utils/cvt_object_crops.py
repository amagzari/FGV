from fastai.vision.all import *
import cv2
from scipy.ndimage import gaussian_filter

def generate_attention_coordinates_cvt(attention_map, 
                                     thresh_method='mean', 
                                     block_size=21, 
                                     method='gaussian', 
                                     min_area=32*32, 
                                     num_bboxes=1, 
                                     random_crop_sz=112):
    """Generate bounding box coordinates from CvT attention maps.
    
    Note: This function is similar to the ViT version but optimized for CvT's 
    attention map characteristics. CvT attention maps come from hierarchical stages,
    so they may have different properties than ViT's uniform attention maps.
    
    Args:
        attention_map: 2D tensor containing attention weights from CvT's last stage
        thresh_method: Method for thresholding ('mean' or 'otsu')
        block_size: Size of block for adaptive thresholding
        method: Method for smoothing ('gaussian' or None)
        min_area: Minimum area for valid bounding boxes
        num_bboxes: Number of bounding boxes to return
        random_crop_sz: Size of random crop within bounding box
        
    Returns:
        list: List of (x1, y1, x2, y2) coordinates for detected objects
    """
    # Convert attention map to numpy and scale to 0-255
    attention_map = attention_map.cpu().numpy()
    attention_map = (attention_map * 255).astype(np.uint8)
    
    # Apply Gaussian smoothing if specified
    if method == 'gaussian':
        attention_map = gaussian_filter(attention_map, sigma=1)
        attention_map = (attention_map * 255).astype(np.uint8)
    
    # Thresholding
    if thresh_method == 'mean':
        thresh = cv2.adaptiveThreshold(
            attention_map,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size,
            0
        )
    else:  # otsu
        _, thresh = cv2.threshold(
            attention_map, 
            0, 
            255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    
    # Find contours
    contours, _ = cv2.findContours(
        thresh, 
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filter contours by area and get bounding boxes
    valid_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            valid_boxes.append([x, y, x+w, y+h])
    
    # Sort boxes by area (largest first)
    valid_boxes.sort(key=lambda box: (box[2]-box[0])*(box[3]-box[1]), reverse=True)
    
    # Return requested number of boxes
    return valid_boxes[:num_bboxes] if valid_boxes else [] 