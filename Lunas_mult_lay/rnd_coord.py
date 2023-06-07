# --------------------------------------------------- #
# File: rnd_coord.py
# Author: Mariana Zamudio Ayala
# Date: 17/05/2023
# --------------------------------------------------- #
import random
import math
import matplotlib.pyplot as plt
import numpy as np

def generate_coords_in_moon(r, w, d, num_coord, region='A'):
    """
        Function that obtains random coordinates within a moon
        area defined by its radious, width and distance from x1-axis
        when its located in region B. 

        Parameters
        ----------
        r : int of float
            moon radious
        w : int or float
            moon width
        d : int or float
        vertical distance of the moon center from x1 axis
            (only applicable for moons in region B)
        num_coord : int
            number of coordinates created
            region: A ---> quadrants I and II of cartesian plane (moon center in (0,0))
                    B ---> quadrants III and IV of cartesian plane (moon center in (r,-d))
    """
    # Initialize variables depending on the region of the moon
    if region =='A':
        center = (0,0)
        a = 1
    elif region == 'B':
        center = (r,-d)
        a = -1
    
    # Define max and min radious of the moon
    rad_min = r - w/2
    rad_max = r + w/2
    
    # Iterate to create each coordinate
    for i in range(num_coord):
        # Choose a random radious and angle between the stablished limits
        r = random.uniform(rad_min, rad_max) 
        theta = random.uniform(0, a*math.pi)

        # Transform polar coordinates to cartesian coordinates
        x = r*math.cos(theta) + center[0]
        y = r*math.sin(theta) + center[1]
        
        # Add coordinates to the list
        if i == 0:
            coords = np.array([[x], [y]])
        else:
            temp = np.array([[x], [y]])
            coords = np.hstack((coords,temp))


    return (coords)

if __name__ == "__main__":
    # --------------Test
    # Center radious of the moon 
    r = 3  
    # Vertical distance between center of moon 
    # in region A and moon in region B
    d = 0
    # width of the moons 
    w = 1  

    # Obtener coordenadas en region A
    coords_A = generate_coords_in_moon(r, w, d, 500)

    # Obtener coordenadas en region B
    coords_B= generate_coords_in_moon(r,w,d,500, "B")

    # Graficar coordenadas
    plt.scatter(coords_A[0,:], coords_A[1,:])
    plt.scatter(coords_B[0,:], coords_B[1,:], marker="x")
    plt.grid(True)
    plt.show()