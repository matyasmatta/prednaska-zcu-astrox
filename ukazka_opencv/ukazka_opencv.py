from exif import Image
from datetime import datetime
import cv2
import math
import numpy as np
import statistics
import os

# This is code for better calculation of position of north on photo
# We did analyze on example data which were on raspberry and found that compass can be easy affected by other magnetic fields. The difference between the correct position of north and the data from compass were sometimes different by 30 degrees
# So we invented equation that describe position of north towards positition of ISS (the ISS is looking in the direction of its flight).
# Then we are calculating relative rotation of a camera towards the ISS with opencv . We are tracking the movement of picture on compering it to the movement of the ISS on then calculating the relative rotation of camera on ISS


#for acessing program via other .py file
def find_north(image_1, image_2):
    number_of_displaying_matches = int(input("Přesnost:"))

    #geting EXIF time of capture
    def get_time(image):
        with open(image, 'rb') as image_file:
            img = Image(image_file)
            try:
                time_str = img.get("datetime_original")
                time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
            except TypeError:
                time = 0
        return time
    
    #converting images to cv friendly readable format 
    def convert_to_cv(image_1, image_2):
        image_1_cv = cv2.imread(image_1, 0)
        image_2_cv = cv2.imread(image_2, 0)
        return image_1_cv, image_2_cv

    #finding same "things" on both images
    def calculate_features(image_1, image_2, feature_number):
        orb = cv2.ORB_create(nfeatures = feature_number)
        keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
        keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)
        return keypoints_1, keypoints_2, descriptors_1, descriptors_2

    #connecting same "things" on photo
    def calculate_matches(descriptors_1, descriptors_2):
        brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = brute_force.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
    
    #displaying the matches (only works on PC)
    def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches):
        match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches[:10000000], None)
        resize = cv2.resize(match_img, (1600,600), interpolation = cv2.INTER_AREA)
        cv2.imshow('matches', resize)
        cv2.waitKey(0)
        cv2.destroyWindow('matches')

    #finding coordination of same "things" on both fotos
    def find_matching_coordinates(keypoints_1, keypoints_2, matches):
        coordinates_1 = []
        coordinates_2 = []
        global x_1all_div
        global x_2all_div
        global y_1all_div
        global y_2all_div
        x_11all= []
        x_22all= []
        y_11all= []
        y_22all= []
        for match in matches:
            image_1_idx = match.queryIdx
            image_2_idx = match.trainIdx
            (x1,y1) = keypoints_1[image_1_idx].pt
            (x2,y2) = keypoints_2[image_2_idx].pt
            coordinates_1.append((x1,y1))
            coordinates_2.append((x2,y2))
            #we store all matched coordinates to list for further calculation
            x_11all.append(x1)
            x_22all.append(x2)
            y_11all.append(y1)
            y_22all.append(y2)
        #this calculates us the median of all coordinations on output [x1, y1] and [x2,y2]
        x_11all_div=0
        x_11all_div=statistics.median(x_11all)
        x_22all_div=0
        x_22all_div=statistics.median(x_22all)
        y_11all_div=0
        y_11all_div=statistics.median(y_11all)
        y_22all_div=0
        y_22all_div=statistics.median(y_22all)
        
        #we find the vector of median coordinates and place them into one of four quadrants 
        global direction_x
        global direction_y
        delta_x = x_11all_div-x_22all_div
        if delta_x > 0:
            direction_x = "left"
        elif delta_x < 0:
            direction_x = "right"
        else: 
            direction_x = "null"
        delta_y = y_11all_div-y_22all_div
        if delta_y > 0:
            direction_y = "up"
        elif delta_y < 0:
            direction_y = "down"
        else:
            direction_y = "null"

        #we calculate the angle of movemment of "things" on photo
        delta_x = abs(delta_x)
        delta_y = abs(delta_y)
        tangens_angle_for_general_direction_radians = np.arctan((delta_y)/(delta_x))
        tangens_angle_for_general_direction_degrees = tangens_angle_for_general_direction_radians * (360/(2*np.pi))

        return coordinates_1, coordinates_2, tangens_angle_for_general_direction_degrees
    
    #getting latitude of both images from EXIF data
    def get_latitude(image):
        with open(image, 'rb') as image_file:
            img = Image(image_file)
            if img.has_exif:
                try:
                    latitude = img.get("gps_latitude")
                    latitude_ref = img.get("gps_latitude_ref")
                    if latitude == None:
                        latitude, latitude_ref = (0.0, 0.0, 0.0), "A"
                except AttributeError:
                    latitude, latitude_ref = (0.0, 0.0, 0.0), "A"
            else:
                latitude, latitude_ref = (0.0, 0.0, 0.0), "A"
        return latitude, latitude_ref
    
    #converting latitude to decimal
    def get_decimal_latitude(latitude, latitude_ref):
        decimal_degrees = latitude[0] + latitude[1] / 60 + latitude[2] / 3600
        if latitude_ref == "S" or latitude_ref == "W":
            decimal_degrees = -decimal_degrees
        return decimal_degrees

    #getting latitude for using
    def get_latitudes(image_1, image_2):    
        latitude_image_1_x, latitude_image_1_ref = get_latitude(image_1)
        latitude_image_1 = get_decimal_latitude(latitude_image_1_x, latitude_image_1_ref)
        latitude_image_2_x, latitude_image_2_ref = get_latitude(image_2)
        latitude_image_2 = get_decimal_latitude(latitude_image_2_x, latitude_image_2_ref)
        return latitude_image_1, latitude_image_2

    #using defined functions
    latitude_image_1, latitude_image_2 = get_latitudes(image_1, image_2)
    image_1_cv, image_2_cv = convert_to_cv(image_1, image_2) 
    keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, number_of_displaying_matches) 
    matches = calculate_matches(descriptors_1, descriptors_2)
    display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches)
    coordinates_1, coordinates_2, tangens_angle_for_general_direction_degrees = find_matching_coordinates(keypoints_1, keypoints_2, matches)

    tangens_angle_for_general_direction_degrees = abs(tangens_angle_for_general_direction_degrees)

    #calculating the relative rotation of camera on ISS
    edoov_coefficient = ""
    if direction_x == "left":
        if direction_y == "up":
            edoov_coefficient = (tangens_angle_for_general_direction_degrees, -1, -1, "↖")
            clockwise_edoov_coefficient = 270-tangens_angle_for_general_direction_degrees
        if direction_y == "down":
            edoov_coefficient = (tangens_angle_for_general_direction_degrees, -1, 1,"↙")
            clockwise_edoov_coefficient = 270+tangens_angle_for_general_direction_degrees
    if direction_x == "right":
        if direction_y == "up":
            edoov_coefficient = (tangens_angle_for_general_direction_degrees, 1, -1, "↗")
            clockwise_edoov_coefficient = 90+tangens_angle_for_general_direction_degrees
        if direction_y == "down":
            edoov_coefficient = (tangens_angle_for_general_direction_degrees, 1, 1, "↘")
            clockwise_edoov_coefficient = 90-tangens_angle_for_general_direction_degrees
    print(edoov_coefficient[3])

if __name__ == '__main__':
    find_north(r"C:\Users\trajc\OneDrive\Dokumenty\Python_scripts\timatable_scraper\prednaska-zcu-astrox\ukazka_opencv\mad1.jpg", r"C:\Users\trajc\OneDrive\Dokumenty\Python_scripts\timatable_scraper\prednaska-zcu-astrox\ukazka_opencv\mad2.jpg")