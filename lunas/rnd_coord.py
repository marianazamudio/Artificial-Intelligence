# --------------------------------------------------- #
# File: rnd_coord.py
# Author: Mariana Zamudio Ayala
# Date: 17/05/2023
# Description: Function that generates random coor-
# dinates inside a moon of half ring figure defined by:
# * the region type:
#       * A: with its center in (0,0), located in the upper
#            cuadrants (I y II)
#       * B: with its center in (r, -d), located usually in
#            the inferior cuadrants (III, VI) when d is
#            positive.
# * radius
# * width
# * distance (for the type B in the iferior cuadrants)
# NOTE: This is not the better or most efficient approach
# --------------------------------------------------- #
import random
import math
import matplotlib.pyplot as plt

def generate_coords_in_moon(r, w, d, num_coord, region='A'):
    # Obtain limits in of regions in axis x and y
    # Limits form rectangular zones
    if region =='A':
        limits_x = (-r-w/2, r+w/2)
        limits_y = (0, r+w/2)
        center = (0,0)
    elif region == 'B':
        limits_x = (-w/2, 2*r+w/2)
        limits_y = (-d-r-w/2, -d)
        center = (r,-d)
    
    counter = 0
    coords_x = []
    coords_y = []
    print(counter, num_coord)
    
    while counter < num_coord:
        x = random.uniform(limits_x[0], limits_x[1])  # Generate a random value for x
        y = random.uniform(limits_y[0], limits_y[1])  # Generate a random value for y

        # ---Verify if the point is inside the moon
        # Obtain distance between center and the rdm point
        distance = math.sqrt((x-center[0])**2 + (y-center[1])**2)
        
        # Based on the distance verify if the rdm point is inside the moon
        if (r-w/2 <= distance <= r+w/2):
            coords_x.append(x)
            coords_y.append(y)
            counter += 1
        print(coords_x)
    return (coords_x, coords_y)

# --------------Test
r = 3  # Radio interior de la luna menor
d = 2  # Distancia vertical entre los centros de las lunas
w = 1  # Anchura de cada luna

# Obtener coordenadas en region A
coords_xA, coords_yA = generate_coords_in_moon(r, w, d, 500)

# Obtener coordenadas en region B
coords_xB, coords_yB = generate_coords_in_moon(r,w,d,500, "B")

# Graficar coordenadas
plt.scatter(coords_xA, coords_yA)
plt.scatter(coords_xB, coords_yB, marker="x")
plt.grid(True)
plt.show()