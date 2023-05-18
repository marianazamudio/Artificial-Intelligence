import random
import math
import matplotlib.pyplot as plt

def generate_coords_in_moon(r, w, d, num_coord, region='A'):
    # Obtain centers 
    # TODO
    if region =='A':
        center = (0,0)
        a = 1
    elif region == 'B':
        center = (r,-d)
        a = -1
    
    rad_min = r - w/2
    rad_max = r + w/2
    print(rad_min)
    print(rad_max)
    coords_x = []
    coords_y = []
    
    for i in range(num_coord):
        r = random.uniform(rad_min, rad_max) 
        theta = random.uniform(0, a*math.pi)

        x = r*math.cos(theta) + center[0]
        y = r*math.sin(theta) + center[1]

        #print(r)
        #print(theta)
        #print(x)
        #print(y)
        
        coords_x.append(x)
        coords_y.append(y)

    return (coords_x, coords_y)

# --------------Test
r = 3  # Radio interior de la luna menor
d = -2  # Distancia vertical entre los centros de las lunas
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