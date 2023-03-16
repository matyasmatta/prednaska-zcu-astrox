from PIL import Image
import numpy as np
from numpy import average 
from skyfield import api
from skyfield import almanac
import os
import json
import csv
import threading
import time
from time import sleep
import cv2
import numpy as np
from numpy import average 
import statistics
from PIL import Image, ImageStat
from skyfield import api
from skyfield.api import load
import csv
from csv import writer
from datetime import timedelta, datetime
from pathlib import Path
from exif import Image as exify
import os

class shadow:
    # this function is used as for printing data (first used withing shadow so for legacy kept here)
    def print_log(data):
        with open('log.txt', 'a') as f:
            f.write(data)
            f.write("\n")

    # this class gets coordinates from EXIF and converts them into a more friendly decimal format (not to be confused with photo.convert())
    class coordinates:
        def get_latitude(image):
            with open(image, 'rb') as image_file:
                img = exify(image_file)
                try:
                    latitude = img.get("gps_latitude")
                    latitude_ref = img.get("gps_latitude_ref")
                    if latitude == None:
                        latitude, latitude_ref = (0.0, 0.0, 0.0), "A"
                except AttributeError:
                    latitude, latitude_ref = (0.0, 0.0, 0.0), "A"
            decimal_degrees = latitude[0] + latitude[1] / 60 + latitude[2] / 3600
            latitude_formatted = str(str(decimal_degrees)+" "+str(latitude_ref))
            return latitude_formatted

        def get_longitude(image):
            with open(image, 'rb') as image_file:
                img = exify(image_file)
                try:
                    longitude = img.get("gps_longitude")
                    longitude_ref = img.get("gps_longitude_ref")
                    if longitude == None:
                        longitude, longitude_ref = (0.0, 0.0, 0.0), "A"
                except AttributeError:
                    longitude, longitude_ref = (0.0, 0.0, 0.0), "A"
            decimal_degrees = longitude[0] + longitude[1] / 60 + longitude[2] / 3600
            longitude_formatted = str(str(decimal_degrees)+" "+longitude_ref)
            return longitude_formatted   

    # here we define a subclass containing all we need to calculate sun data, i.e. altitude and azimuth, both can be triggered separately
    class sun_data:
        def altitude(coordinates_latitude, coordinates_longtitude, year, month, day, hour, minute, second):
            # use the NASA API to be able to calculate sun's position
            ts = api.load.timescale()
            ephem = api.load("ftp://ssd.jpl.nasa.gov/pub/eph/planets/bsp/de421.bsp")

            # define sky objects
            sun = ephem["Sun"]
            earth = ephem["Earth"]

            # given coordinates calculate the altitude (how many degrees sun is above the horizon), additional data is redundant
            location = api.Topos(coordinates_latitude, coordinates_longtitude, elevation_m=500)
            sun_pos = (earth + location).at(ts.tt(year,month,day,hour,minute,second)).observe(sun).apparent()
            altitude, azimuth, distance = sun_pos.altaz()
            altitude= float(altitude.degrees)
            return(altitude)
        def azimuth(coordinates_latitude, coordinates_longtitude, year, month, day, hour, minute, second):
            # use the NASA API to be able to calculate sun's position
            ts = api.load.timescale()
            ephem = api.load("ftp://ssd.jpl.nasa.gov/pub/eph/planets/bsp/de421.bsp")

            # define sky objects
            sun = ephem["Sun"]
            earth = ephem["Earth"]

            # given coordinates calculate the altitude (how many degrees sun is above the horizon), additional data is redundant
            location = api.Topos(coordinates_latitude, coordinates_longtitude, elevation_m=500)
            sun_pos = (earth + location).at(ts.tt(year,month,day,hour,minute,second)).observe(sun).apparent()
            altitude, azimuth, distance = sun_pos.altaz()
            azimuth= float(azimuth.degrees)
            return(azimuth)

    # because the brightest point needn't be the centre we ofset the starting pixel by a number defined partially by a constant 
    def starting_point_corrector(x_centre, y_centre, x_increase_final, y_increase_final):
        global constant_for_starting_point_correction
        constant_for_starting_point_correction = 10
        x_final = x_centre - constant_for_starting_point_correction*x_increase_final
        y_final = y_centre - constant_for_starting_point_correction*y_increase_final
        x_final = round(x_final, 0)
        x_final = int(x_final)
        y_final = round(y_final, 0)
        y_final = int(y_final)
        return x_final, y_final
    
    # here we calculate the angle used to seatch for shadows (it is mostly just formatting now as of v4.4)
    def calculate_angle_for_shadow(latitude, longitude, year, month, day, hour=0, minute=0, second=0):
        azimuth = shadow.sun_data.azimuth(latitude, longitude, year, month, day, hour, minute, second)
        total_angle = azimuth + 180
        while total_angle >= 360:
            total_angle -= 360
        return total_angle

    # this is the most important function of the whole class, it calculates cloud height knowing only coordinates
    # we use robust error handling to make sure the function does not cause a fatal error, it is a very complicated function
    def calculate_shadow(x, y, angle, cloud_id="not specified", image_id="not specified", file_path="not specified", image_direct="not specified"):
        try:
            im = Image.open(file_path) # Can be many different formats.
            pix = im.load()

            # get the width and height of the image for iterating over
            total_x, total_y = im.size

            # the next lines of code are very complex, but the method is as follows
            # 1) we need to find "how does x increase in regards to y and vice versa?"
            # 2) we need that either x or y are set as a constant 1 whereas the other is used as addition to a sum
            # 3) when the sum overflows we move a pixel in the less significant direction
            # because of comparasions we need to work with absolute values, basic angles and sectors
            # we are aware that there might be a simpler solution but this is the only one we found consistent and fairly fast

            # calculate meta angle
            angle_radians =np.radians(angle)
            x_increase_meta = np.sin(angle_radians)
            y_increase_meta = np.cos(angle_radians)
            y_increase_meta = -y_increase_meta
            x_increase_meta = np.round(x_increase_meta,5)
            y_increase_meta = np.round(y_increase_meta,5)
            x_increase_meta_abs = abs(x_increase_meta)
            y_increase_meta_abs = abs(y_increase_meta)

            # divide into quadrant and basic angle information
            # quadrant is used for pixel by pixel fingerprint and angle_final is used for final Pythagorian correction
            if 0 <= angle <= 90:
                q = 1
                angle_final = angle 
            if 90 < angle <= 180:
                q = 2
                angle_final = angle - 90
            if 180 < angle <= 270:
                q = 3
                angle_final = angle - 180
            if 270 < angle <= 360:
                q = 4
                angle_final = angle - 270

            # make at least one variable 1 and the other smaller than 1
            if x_increase_meta_abs > y_increase_meta_abs:
                x_increase_final = 1
                y_increase_final = y_increase_meta_abs/x_increase_meta_abs
            if x_increase_meta_abs == y_increase_meta_abs:
                x_increase_final = 1
                y_increase_final = 1
            if x_increase_meta_abs < y_increase_meta_abs:
                x_increase_final = x_increase_meta_abs/y_increase_meta_abs
                y_increase_final = 1
            if angle_final == 0:
                x_increase_final = 0
                y_increase_final = 1

            # set absolute final values
            x_increase_final_abs = abs(x_increase_final)
            y_increase_final_abs = abs(y_increase_final)

            # set values for future reference
            y_sum = 0
            x_sum = 0
            count = -1
            list_of_values = []
            list_of_red = []

            # automatic limit calculation
            # here we set basically how far we should search for shadows
            sun_altitude_for_limit = shadow.sun_data.altitude("34.28614 S", "147.9849 E", 2022, 1, 15, 5, 16, 5)
            sun_altitude_for_limit_radians = sun_altitude_for_limit*(np.pi/180)
            limit_cloud_height = 12000 #meters
            limit_shadow_cloud_distance = limit_cloud_height/np.tan(sun_altitude_for_limit_radians)
            limit_shadow_cloud_distance_pixels = limit_shadow_cloud_distance/126.48
            limit = limit_shadow_cloud_distance_pixels

            if os.path.exists('meta.jpg') == False:
                im2 = im.copy()
                im.save('meta.jpg')
            # put quarter information back for pixel reading
            ## first two lines are used for limit setting
            ## please note that explanation for all quarters are very similar to quarter 1, hence please excuse that we did not write the documentation for every sector
            ## we apologise for repetitive code, but it works perfectly (which was remarkably difficult)
            if q == 1:
                x_increase_final = x_increase_final
                y_increase_final = -y_increase_final
                x,y = shadow.starting_point_corrector(x,y, x_increase_final, y_increase_final)
                while True:
                    if x_increase_final_abs > y_increase_final_abs:
                        # check if y_sum is bigger than 1
                        y_sum = abs(y_sum)
                        if y_sum >= 1:
                            y_sum -= 1
                            y -= 1
                        # read pixel value
                        data = (pix[x,y])
                        # print(data)
                        value = round(average(data))
                        value_red = data[0]
                        # print(value)
                        # print("red", value_red)
                        list_of_red.append(value_red)
                        list_of_values.append(value)
                        im2 = Image.open('meta.jpg')
                        im2.putpixel((x,y),(0,0,0,0))
                        im2.save('meta.jpg')
                        # add to y_sum and move pixel x for 1
                        x += 1
                        y_sum += y_increase_final_abs
                        # print(count, limit)
                        if x > total_x or y > total_y:
                            break
                    if x_increase_final_abs == y_increase_final_abs:               
                        data = (pix[x,y])
                        # print(data)
                        value = round(average(data))
                        value_red = data[0]
                        # print(value)
                        # print("red", value_red)
                        list_of_red.append(value_red)
                        list_of_values.append(value)
                        im2 = Image.open('meta.jpg')
                        im2.putpixel((x,y),(0,0,0,0))
                        im2.save('meta.jpg')
                        x += 1
                        y -= 1
                        # print(count, limit)
                        if x > total_x or y > total_y:
                            break
                    if x_increase_final_abs < y_increase_final_abs:               
                        x_sum = abs(x_sum)
                        if x_sum >= 1:
                            x_sum -= 1
                            x += 1
                        # read pixel value
                        data = (pix[x,y])
                        # print(data)
                        value = round(average(data))
                        value_red = data[0]
                        # print(value)
                        # print("red", value_red)
                        list_of_red.append(value_red)
                        list_of_values.append(value)
                        im2 = Image.open('meta.jpg')
                        im2.putpixel((x,y),(0,0,0,0))
                        im2.save('meta.jpg')
                        # add to y_sum and move pixel x for 1
                        y -= 1
                        x_sum += x_increase_final_abs
                        # print(count, limit)
                        if x > total_x or y > total_y:
                            break
                    #write into txt
                    with open('stiny.txt', 'a') as f:
                        value = str(value)
                        f.write(value)
                        f.write("\n")
                    with open('stiny_red.txt', 'a') as f:
                        value_red = str(value_red)
                        f.write(value_red)
                        f.write("\n")
                    if count > limit:
                        break
            if q == 2:
                x_increase_final = x_increase_final
                y_increase_final = y_increase_final  
                x,y = shadow.starting_point_corrector(x,y, x_increase_final, y_increase_final) 
                while True:
                    count += 1
                    if x_increase_final_abs > y_increase_final_abs:        
                        # check if y_sum is bigger than 1
                        y_sum = abs(y_sum)
                        if y_sum >= 1:
                            y_sum -= 1
                            y += 1
                        # read pixel value
                        data = (pix[x,y])
                        # print(data)
                        value = round(average(data))
                        value_red = data[0]
                        # print(value)
                        # print("red", value_red)
                        list_of_red.append(value_red)
                        list_of_values.append(value)
                        im2 = Image.open('meta.jpg')
                        im2.putpixel((x,y),(0,0,0,0))
                        im2.save('meta.jpg')
                        # add to y_sum and move pixel x for 1
                        x += 1
                        y_sum += y_increase_final_abs
                        # print(count, limit)
                        if x > total_x or y > total_y:
                            break
                    if x_increase_final_abs == y_increase_final_abs:          
                        data = (pix[x,y])
                        # print(data)
                        value = round(average(data))
                        value_red = data[0]
                        # print(value)
                        # print("red", value_red)
                        list_of_red.append(value_red)
                        list_of_values.append(value)
                        im2 = Image.open('meta.jpg')
                        im2.putpixel((x,y),(0,0,0,0))
                        im2.save('meta.jpg')
                        x += 1
                        y += 1
                        # print(count, limit)
                        if x > total_x or y > total_y:
                            break
                    if x_increase_final_abs < y_increase_final_abs:   
                        x_sum = abs(x_sum)
                        if x_sum >= 1:
                            x_sum -= 1
                            x += 1
                        # read pixel value
                        data = (pix[x,y])
                        # print(data)
                        value = round(average(data))
                        value_red = data[0]
                        # print(value)
                        # print("red", value_red)
                        list_of_red.append(value_red)
                        list_of_values.append(value)
                        im2 = Image.open('meta.jpg')
                        im2.putpixel((x,y),(0,0,0,0))
                        im2.save('meta.jpg')
                        # add to y_sum and move pixel x for 1
                        y += 1
                        x_sum += x_increase_final_abs
                        #write into txt
                        # print(count, limit)
                        if x > total_x or y > total_y:
                            break
                    with open('stiny.txt', 'a') as f:
                        value = str(value)
                        f.write(value)
                        f.write("\n")
                    with open('stiny_red.txt', 'a') as f:
                        value_red = str(value_red)
                        f.write(value_red)
                        f.write("\n")
                    if count > limit:
                        break
            if q == 3:
                x_increase_final = -x_increase_final
                y_increase_final = y_increase_final 
                x,y = shadow.starting_point_corrector(x,y, x_increase_final, y_increase_final)
                while True:
                    count += 1
                    if x_increase_final_abs > y_increase_final_abs:
                        # check if y_sum is bigger than 1
                        y_sum = abs(y_sum)
                        if y_sum >= 1:
                            y_sum -= 1
                            y += 1
                        # read pixel value
                        data = (pix[x,y])
                        # print(data)
                        value = round(average(data))
                        value_red = data[0]
                        # print(value)
                        # print("red", value_red)
                        list_of_red.append(value_red)
                        list_of_values.append(value)
                        im2 = Image.open('meta.jpg')
                        im2.putpixel((x,y),(0,0,0,0))
                        im2.save('meta.jpg')
                        # add to y_sum and move pixel x for 1
                        x -= 1
                        y_sum += y_increase_final_abs
                        # print(count, limit)
                        if x > total_x or y > total_y:
                            break
                    if x_increase_final_abs == y_increase_final_abs:
                        data = (pix[x,y])
                        # print(data)
                        value = round(average(data))
                        value_red = data[0]
                        # print(value)
                        # print("red", value_red)
                        list_of_red.append(value_red)
                        list_of_values.append(value)
                        im2 = Image.open('meta.jpg')
                        im2.putpixel((x,y),(0,0,0,0))
                        im2.save('meta.jpg')
                        x -= 1
                        y += 1
                        # print(count, limit)
                        if x > total_x or y > total_y:
                            break
                    if x_increase_final_abs < y_increase_final_abs:
                        x_sum = abs(x_sum)
                        if x_sum >= 1:
                            x_sum -= 1
                            x -= 1
                        # read pixel value
                        data = (pix[x,y])
                        # print(data)
                        value = round(average(data))
                        value_red = data[0]
                        # print(value)
                        # print("red", value_red)
                        list_of_red.append(value_red)
                        list_of_values.append(value)
                        im2 = Image.open('meta.jpg')
                        im2.putpixel((x,y),(0,0,0,0))
                        im2.save('meta.jpg')
                        # add to y_sum and move pixel x for 1
                        y += 1
                        x_sum += x_increase_final_abs
                        # print(count, limit)
                        if x > total_x or y > total_y:
                            break
                    #write into txt
                    value = str(value)
                    with open('stiny.txt', 'a') as f:
                        f.write(value)
                        f.write("\n")
                    if count > limit:
                        break
            if q == 4:
                x_increase_final = -x_increase_final
                y_increase_final = -y_increase_final 
                x,y = shadow.starting_point_corrector(x,y, x_increase_final, y_increase_final)
                while True:
                    count += 1
                    if x_increase_final_abs > y_increase_final_abs:
                        # check if y_sum is bigger than 1
                        y_sum = abs(y_sum)
                        if y_sum >= 1:
                            y_sum -= 1
                            y -= 1
                        # read pixel value
                        data = (pix[x,y])
                        # print(data)
                        value = round(average(data))
                        value_red = data[0]
                        # print(value)
                        # print("red", value_red)
                        list_of_red.append(value_red)
                        list_of_values.append(value)
                        im2 = Image.open('meta.jpg')
                        im2.putpixel((x,y),(0,0,0,0))
                        im2.save('meta.jpg')
                        # add to y_sum and move pixel x for 1
                        x -= 1
                        y_sum += y_increase_final_abs
                        # print(count)
                        # print(count, limit)
                        if x > total_x or y > total_y:
                            break
                    if x_increase_final_abs == y_increase_final_abs:
                        data = (pix[x,y])
                        # print(data)
                        value = round(average(data))
                        value_red = data[0]
                        # print(value)
                        # print("red", value_red)
                        list_of_red.append(value_red)
                        list_of_values.append(value)
                        im2 = Image.open('meta.jpg')
                        im2.putpixel((x,y),(0,0,0,0))
                        im2.save('meta.jpg')
                        x -= 1
                        y -= 1
                        # print(count, limit)
                        if x > total_x or y > total_y:
                            break
                    if x_increase_final_abs < y_increase_final_abs:
                        x_sum = abs(x_sum)
                        if x_sum >= 1:
                            x_sum -= 1
                            x -= 1
                        # read pixel value
                        data = (pix[x,y])
                        # print(data)
                        value = round(average(data))
                        value_red = data[0]
                        # print(value)
                        # print("red", value_red)
                        list_of_red.append(value_red)
                        list_of_values.append(value)
                        im2 = Image.open('meta.jpg')
                        im2.putpixel((x,y),(0,0,0,0))
                        im2.save('meta.jpg')
                        # add to y_sum and move pixel x for 1
                        y -= 1
                        x_sum += x_increase_final_abs
                        # print(count, limit)
                        if x > total_x or y > total_y:
                            break
                    #write into txt
                    with open('stiny.txt', 'a') as f:
                        value = str(value)
                        f.write(value)
                        f.write("\n")
                    with open('stiny_red.txt', 'a') as f:
                        value_red = str(value_red)
                        f.write(value_red)
                        f.write("\n")
                    if count > limit:
                        break

            # set absolute final values
            x_increase_final_abs = abs(x_increase_final)
            y_increase_final_abs = abs(x_increase_final)

            # define a calculation method for cloud-shadow difference (the following one just takes the min and max values)
            def calculate_using_min_max(list_of_values):
                def main():
                    # find items in list correspoding to the lowest and highest point
                    shadow_low = min(list_of_values)
                    cloud_high = max(list_of_values)

                    # find of said items in the list (their order)
                    shadow_location = list_of_values.index(shadow_low)
                    cloud_location = list_of_values.index(cloud_high)

                    # find the difference
                    shadow_lenght = shadow_location - cloud_location
                    return shadow_lenght, cloud_high, shadow_low, cloud_location, shadow_location
                shadow_lenght, cloud_high, shadow_low, cloud_location, shadow_location = main()
                while True:
                    # in case that shadow is found before a cloud in the line, we delete the value as its false and repeat
                    if shadow_lenght <= 0:
                        list_of_values.remove(cloud_high)
                        shadow_lenght, cloud_high, shadow_low, cloud_location, shadow_location = main()
                    else:
                        break
                return shadow_lenght

            # define a calculation method for cloud-shadow difference (the following one takes the changes in values and their respective min max values)
            def calculate_using_maximum_change(list_of_values):
                def main():
                    n = constant_for_starting_point_correction
                    list_of_changes = []

                    # IMPORTANT: please note that this code will return EXCEPTIONS
                    # it is NEVER a fatal error and IS HANDLED perfectly fine
                    # it was the easiest method as to how to correct for clouds being after shadows (means we detected a different cloud)
                    # the code one by one removes pixels that are detected as the brightest and simultaniously are after the shadow (returns negative lenght)
                    # done as a loop that will return an error when there are no more objects in the list
                    while True:
                        try:
                            current_data = list_of_values[n]
                            previous_data = list_of_values[n-1]
                            change_in_data = current_data-previous_data
                            if n == 0:
                                pass
                            else:
                                list_of_changes.append(change_in_data)
                            n+=1
                        except:
                            break

                    # self-explanatory
                    shadow_low = max(list_of_changes)
                    cloud_high = min(list_of_changes)

                    shadow_location = list_of_changes.index(shadow_low)
                    cloud_location = list_of_changes.index(cloud_high)

                    # find difference between the two pixel lenghts
                    shadow_lenght = shadow_location - cloud_location
                    return shadow_lenght, cloud_high, cloud_location
                
                # first we calculate
                shadow_lenght, cloud_high, cloud_location = main()

                # then we check for clouds after shadows and if necessary re-run the local main function (see above)
                while True:
                    n = constant_for_starting_point_correction
                    if shadow_lenght <= 0:
                        item_to_be_deleted = list_of_values[cloud_location]
                        list_of_values.remove(item_to_be_deleted)
                        shadow_lenght, cloud_high, cloud_location = main()
                    else:
                        break

                return shadow_lenght

            # here we calculate the shadow-cloud distance by each respective method
            shadow_lenght_min_max = calculate_using_min_max(list_of_values)
            shadow_lenght_max_difference = calculate_using_maximum_change(list_of_values)
            shadow_lenght_max_difference_red = calculate_using_maximum_change(list_of_red)

            # we put the methods together
            shadow_lenght_final = (shadow_lenght_max_difference+shadow_lenght_min_max+shadow_lenght_max_difference_red)/3

            # calculate distance based on a distance in pixels
            lenght = int(shadow_lenght_final) * 126.48

            # because original lenghts are adjacent lenghts and not hypotenuse we will have to convert
            angle_final_radians = angle_final*(np.pi/180)
            if angle_final <= 45:
                lenght = lenght/np.cos(angle_final_radians)
            if angle_final > 45:
                lenght = lenght/np.sin(angle_final_radians)

            # now we calculate the sun altitude using a function
            altitude = shadow.sun_data.altitude("34.28614 S", "147.9849 E", 2022, 1, 15, 5, 16, 5)

            # calculate final values
            altitude_radians = altitude*(np.pi/180)
            cloudheight = np.tan(altitude_radians)*lenght
            cloudheight = np.round(cloudheight,2)
        except:
            # sometimes the AI models returns negative integers, these break the code and are difficult to protect against
            # so in case of an error we pass it to the outer function as well and make sure that the whole shadow process does not crash
            cloudheight = "error"
        return cloudheight    

if __name__ == '__main__':
    cloudheight = shadow.calculate_shadow(file_path='zchop.meta.x000.y000.n011.jpg', x=294,y=199,angle=310)
    print(cloudheight)
