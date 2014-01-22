# -*- coding:utf-8 -*-
import numpy as np

class Obstacle:
    def __init__(self, ox, oy):
        self.ox = ox
        self.oy = oy

class Rectangle (Obstacle):
    def __init__(self, ox, oy, w, h):
        self.ox = ox
        self.oy = oy
        self.width = w
        self.height = h
        self.posX = ox -1 # Utile pour incrémenter au début
        self.posY = oy
        self.theta = 0
    def __iter__ (self):
        return self
    def next(self):
        # On essaie d'avancer sur la ligne
        if self.posX < self.width:
            self.posX = self.posX + 1
            return (self.posX - 1, self.posY)
        # Sinon on essaie de revenir à la ligne
        elif self.posY < self.height:
            # On passe a la ligne
            self.posX = self.ox
            self.posY = self.posY +1
            return (self.posX, self.posY -1)
        else: # On est arrivé au bout
            raise StopIteration

    def rotate(theta) =
        self.theta = theta
# class Circle (Obstacle):
#     def __init__(self, ox, oy, r):
#         self.ox = ox
#         self.oy = oy
#         self.radius = r
#         self.radius2 = r**2
#         # Position dans le cercle, initialement tout en bas
#         self.posX = ox # On va incrémenter au tout début !
#         self.posY = oy-r
#     def __iter__(self):
#         return self
    
#     def next (self):
#         # Si on est dans l'obstacle
#         if (self.posX**2 + self.posY**2) < self.radius2:
#             ret = (self.posX, self.posY)
#             # On essaie d'aller à droite
#             if ((self.posX+1)**2 + self.posY**2) < self.radius2:
#                 self.posX += 1
#             # On essaie de monter, si possible on met alors x² = r²-y²
#             elif (self.posX**2 + (self.posY+1)**2) < self.radius2:
#                 self.posY += 1
#                 self.posX = -int(np.sqrt(self.radius2 - self.posY**2))
#             return ret

#         else:
#             raise StopIteration

a = Circle(0,0,5)
b = Rectangle(0,0,2,3)
for i,j in a:
    print (i,j)
