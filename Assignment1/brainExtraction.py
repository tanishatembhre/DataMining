import os
import cv2
import numpy as np
import shutil
import logging

logger = logging.getLogger(__file__)
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)


def slice_contour_brain(image_folder, output_dir=""):
    logger.info(f"Finding thresh files in {os.path.abspath(image_folder)}")
    logger.info(f"Output directory :{os.path.abspath(output_dir)}")
    slicedir = os.path.join(output_dir, "Slices")
    if os.path.exists(slicedir):
        shutil.rmtree(slicedir, ignore_errors=False, onerror=None)
    os.makedirs(slicedir)
    boundarydir = os.path.join(output_dir, "Boundaries")
    if os.path.exists(boundarydir):
        shutil.rmtree(boundarydir, ignore_errors=False, onerror=None)
    os.makedirs(boundarydir)
    image_count = 0
    for file in os.listdir(image_folder):
        if file.endswith("thresh.png"):
            image_count += 1
            slice_dir = os.path.join(slicedir, file[:-len("thresh.png") - 1])
            os.makedirs(slice_dir)

            boundary_dir = os.path.join(boundarydir, file[:-len("thresh.png") - 1])
            os.makedirs(boundary_dir)

            # Read brain image and convert to grayscale
            brain_rgb = cv2.imread(os.path.join(image_folder, file))
            brain_gray = cv2.cvtColor(brain_rgb, cv2.COLOR_BGR2GRAY)

            # Read template and convert to grayscale
            template = cv2.imread("r.jpg")
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            # Do template matching
            res = cv2.matchTemplate(brain_gray, template, cv2.TM_CCOEFF_NORMED)

            # Threshold for template matching
            threshold = 0.8
            margin = {'top': 8, 'bottom': 0, 'left': 0, 'right': 0}
            # Storing coordinates of matched Rs in numpy array
            r_array = np.where(res >= threshold)
            coordinates_array = list(zip(*r_array[::-1]))
            slice_dimension = coordinates_array[1][0] - coordinates_array[0][0]
            
            for im_index, pt in enumerate(zip(*r_array[::-1])):
                # Brain slices obtained by cropping
                cropped_img = brain_rgb[pt[1] + margin['top']: pt[1] + slice_dimension - margin['bottom'],
                              pt[0] + margin['left']: pt[0] + slice_dimension - margin['right']]
                cropped_gray = brain_gray[pt[1] + margin['top']: pt[1] + slice_dimension - margin['bottom'],
                              pt[0] + margin['left']: pt[0] + slice_dimension - margin['right']]

                # Finding boundaries in brain slices
                ret, thresh = cv2.threshold(cropped_gray, 0, 255, cv2.THRESH_BINARY)
                edges, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # Output brain slices and brain boundaries only if contours are detected i.e. image is not blank
                if edges:
                    boundaries_img = cropped_img.copy()
                    cv2.drawContours(boundaries_img, edges, -1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imwrite(os.path.join(slice_dir, str(im_index)+".jpg"), cropped_img)
                    cv2.imwrite(os.path.join(boundary_dir, str(im_index) + ".jpg"), boundaries_img)

    logger.info(f"Total images processed : {image_count}")



