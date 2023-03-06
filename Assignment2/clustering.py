import os
import cv2
import numpy as np
import shutil
import logging
import csv
from sklearn.cluster import DBSCAN
from collections import Counter

logger = logging.getLogger(__file__)
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)


def slice_cluster_brain(image_folder, output_dir=""):
    logger.info(f"Finding thresh files in {os.path.abspath(image_folder)}")
    logger.info(f"Output directory :{os.path.abspath(output_dir)}")
    slicedir = os.path.join(output_dir, "Slices")
    if os.path.exists(slicedir):
        shutil.rmtree(slicedir, ignore_errors=False, onerror=None)
    os.makedirs(slicedir)
    boundarydir = os.path.join(output_dir, "Clusters")
    if os.path.exists(boundarydir):
        shutil.rmtree(boundarydir, ignore_errors=False, onerror=None)
    os.makedirs(boundarydir)
    image_count = 0
    for file in os.listdir(image_folder):
        if file.endswith("thresh.png"):
            image_count += 1
            slice_dir = os.path.join(slicedir, file[:-len("thresh.png") - 1])
            os.makedirs(slice_dir)

            cluster_dir = os.path.join(boundarydir, file[:-len("thresh.png") - 1])
            os.makedirs(cluster_dir)

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
            cluster_counter = []
            oimg_count=0
            for im_index, pt in enumerate(zip(*r_array[::-1])):
                # Brain slices obtained by cropping
                cropped_img = brain_rgb[pt[1] + margin['top']: pt[1] + slice_dimension - margin['bottom'],
                    pt[0] + margin['left']: pt[0] + slice_dimension - margin['right']]
                # Check google to replace following two lines if image is blank
                cropped_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                if cv2.countNonZero(cropped_gray) != 0:
                    cropped_hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
                    hue, saturation ,sensitivity = cv2.split(cropped_hsv)
                    saturation_threshold = cv2.threshold(saturation, 92, 255, cv2.THRESH_BINARY)[1]
                    saturation_mask = cv2.add(saturation_threshold, 0)
                    highlighted_points = cropped_img.copy()
                    highlighted_points[saturation_mask==0] = 0
                    X, Y, Z = np.where(highlighted_points!=0)
                    cpoints = np.column_stack((X,Y))
                    ans_image = cropped_img.copy()
                    ans_image[::]=0
                    cluster_count=0
                    if cpoints.shape[0]>0:
                        cpoints = np.unique(cpoints, axis=0)
                        #optimise/ confirm eps and min_samples
                        clusters = DBSCAN(eps=2, min_samples=6).fit(cpoints)
                        cluster_point_counts = Counter(clusters.labels_)

                        cluster_to_keep=set()
                        cluster_to_show=set()
                        for key in cluster_point_counts:
                            if cluster_point_counts[key]!=-1:
                                cluster_to_show.add(key)
                            if cluster_point_counts[key]>135:
                                cluster_to_keep.add(key)
                        cluster_count = len(cluster_to_keep)
                        mask = []
                        for label in clusters.labels_:
                            if label in cluster_to_show:
                                mask.append(1)
                            else:
                                mask.append(-1)
                        mask =np.array(mask)
                        cluster_points = cpoints[mask>0]
                        for c in cluster_points:
                            ans_image[c[0]][c[1]] = (0,255,255)
                    oimg_count += 1
                    cv2.imwrite(os.path.join(slice_dir, str(oimg_count)+".jpg"), cropped_img)
                    cv2.imwrite(os.path.join(cluster_dir, str(oimg_count) + ".jpg"), ans_image)
                    cluster_counter.append([oimg_count, cluster_count])
                with open(os.path.join(cluster_dir,  "cluster_count.csv"), "w", encoding="utf-8") as csvfile:
                    fieldnames = ['SliceNumber', 'ClusterCount']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for slicenum, cluster_count  in cluster_counter:
                        writer.writerow({'SliceNumber':slicenum,'ClusterCount':cluster_count})
                    
    logger.info(f"Total images processed : {image_count}")



