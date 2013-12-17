# -*- coding:utf-8 -*-
#
# Convection 2D schema explicite
#      avec points fantomes
#
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lg
import time

###### affichage graphique
import matplotlib.pyplot as plt
plt.ion()
if 'qt' in plt.get_backend().lower():
    try:
        from PyQt4 import QtGui
    except ImportError:
        from PySide import QtGui

                 
def CFL_advection():
    """
    Condition CFL Advection pour dt
    return le nouveau 'dt' tel que

    abs(V) * dt < dx

    dt_new = 0.8*min(dx/umax,dy/vmax)

    """
    precautionADV = 0.8
    umax = max(np.abs(u).max(),0.01) 
    vmax = max(np.abs(v).max(),0.01)
    dt_cfa = precautionADV * min(dx/umax,dy/vmax)

    return dt_cfa

def CFL_explicite():
    """
    Condition CFL pour la diffusion 
    en cas de schema explicite
  
    """
    precautionEXP = 0.3
    
    dt_DeltaU_x = dx**2/(4.*DeltaU)
    dt_DeltaU_y = dy**2/(4.*DeltaU)

    dt_cfl = precautionEXP * min(dt_DeltaU_x,
                                 dt_DeltaU_y)
    return dt_cfl


####
def Advect():
    """
    Calcule la valeur interpolee qui correspond 
    a l'advection a la vitesse au temps n
    
    travaille bien sur le domaine reel [1:-1,1:-1]

    """ 
    global Resu, Resv 

    # Matrice avec des 1 quand on va a droite, 
    # 0 a gauche ou au centre
    Mx2 = np.sign(np.sign(u[1:-1,1:-1]) + 1.)
    Mx1 = 1. - Mx2

    # Matrice avec des 1 quand on va en haut, 
    # 0 en bas ou au centre
    My2 = np.sign(np.sign(v[1:-1,1:-1]) + 1.)
    My1 = 1. - My2

    # Matrices en valeurs absolues pour u et v
    au = abs(u[1:-1,1:-1]) * dt/dx 
    av = abs(v[1:-1,1:-1]) * dt/dy

    # Matrices des coefficients respectivement 
    # central, exterieur, meme x, meme y     
    Cc = (1. - au) * (1. - av) 
    Ce = au * av
    Cmx = (1. - au) * av
    Cmy = (1. - av) * au

    # Calcul des matrices de resultat 
    # pour les vitesses u et v
    Resu[1:-1,1:-1] = (Cc * u[1:-1, 1:-1] +            
                       Ce * (Mx1*My1 * u[2:, 2:] + 
                             Mx1*My2 * u[:-2, 2:] +
                             Mx2*My1 * u[2:, :-2] +
                             Mx2*My2 * u[:-2, :-2]) +  
                       Cmx * (My1 * u[2:, 1:-1] +
                              My2 * u[:-2, 1:-1]) +   
                       Cmy * (Mx1 * u[1:-1, 2:] +
                              Mx2 * u[1:-1, :-2]))
    
    Resv[1:-1,1:-1] = (Cc * v[1:-1, 1:-1] +            
                       Ce * (Mx1*My1 * v[2:, 2:] + 
                             Mx1*My2 * v[:-2, 2:] +
                             Mx2*My1 * v[2:, :-2] +
                             Mx2*My2 * v[:-2, :-2]) +  
                       Cmx * (My1 * v[2:, 1:-1] +
                              My2 * v[:-2, 1:-1]) +   
                       Cmy * (Mx1 * v[1:-1, 2:] +
                              Mx2 * v[1:-1, :-2]))

def BuildLaPoisson():
    """
    pour l'etape de projection
    matrice de Laplacien phi
    avec CL Neumann pour phi

    BUT condition de Neumann pour phi 
    ==> non unicite de la solution

    besoin de fixer la pression en un point 
    pour lever la degenerescence: ici [0][1]
    
    ==> need to build a correction matrix

    """
    ### ne pas prendre en compte les points fantome (-2)
    NXi = nx
    NYi = ny

    ###### Definition of the 1D Lalace operator

    ###### AXE X
    ### Diagonal terms
    dataNXi = [np.ones(NXi), -2*np.ones(NXi), np.ones(NXi)]   
    
    ### Conditions aux limites : Neumann à gauche, rien à droite
    dataNXi[2][1]     = 2.  # SF left
#    dataNXi[0][NXi-2] = 2.  # SF right

    ###### AXE Y
    ### Diagonal terms
    dataNYi = [np.ones(NYi), -2*np.ones(NYi), np.ones(NYi)] 
   
    ### Conditions aux limites : Neumann 
    dataNYi[2][1]     = 2.  # SF low
    dataNYi[0][NYi-2] = 2.  # SF top

    ###### Their positions
    offsets = np.array([-1,0,1])                    
    DXX = sp.dia_matrix((dataNXi,offsets), shape=(NXi,NXi)) * dx_2
    DYY = sp.dia_matrix((dataNYi,offsets), shape=(NYi,NYi)) * dy_2
    #print DXX.todense()
    #print DYY.todense()
    
    ####### 2D Laplace operator
    LAP = sp.kron(sp.eye(NYi,NYi), DXX) + sp.kron(DYY, sp.eye(NXi,NXi))
    
    ####### BUILD CORRECTION MATRIX

    ### Upper Diagonal terms
    dataNYNXi = [np.zeros(NYi*NXi)]
    offset = np.array([1])

    ### Fix coef: 2+(-1) = 1 ==> Dirichlet en un point (redonne Laplacien)
    ### ATTENTION  COEF MULTIPLICATIF : dx_2 si M(j,i) j-NY i-NX
    dataNYNXi[0][1] = -1 * dx_2

    LAP0 = sp.dia_matrix((dataNYNXi,offset), shape=(NYi*NXi,NYi*NXi))
    
    # tmp = LAP + LAP0
    # print LAP.todense()
    # print LAP0.todense()
    # print tmp.todense()
  
    return LAP + LAP0

def ILUdecomposition(LAP):
    """
    return the Incomplete LU decomposition 
    of a sparse matrix LAP
    """
    return  lg.splu(LAP.tocsc(),)


def ResoLap(splu,RHS):
    """
    solve the system

    SPLU * x = RHS

    Args:
    --RHS: 2D array((NY,NX))
    --splu: (Incomplete) LU decomposed matrix 
            shape (NY*NX, NY*NX)

    Return: x = array[NY,NX]
    
    Rem1: taille matrice fonction des CL 

    """
    # array 2D -> array 1D
    f2 = RHS.ravel()

    # Solving the linear system
    x = splu.solve(f2)

    return x.reshape(RHS.shape)

####
def Laplacien(x):
    """
    calcule le laplacien scalaire 
    du champ scalaire x(i,j)
    
    pas de termes de bord car ghost points

    """
    rst = np.empty((NY,NX))

    coef0 = -2*(dx_2 + dy_2) 
    
    rst[1:-1,1:-1] = ( (x[1:-1, 2:] + x[1:-1, :-2])*dx_2 +  
                       (x[2:, 1:-1] + x[:-2, 1:-1])*dy_2 +  
                       (x[1:-1, 1:-1])*coef0 )    
    return rst

def divergence(u,v):
    """
    divergence avec points fantomes
    ne jamais utiliser les valeurs au bord

    """
    tmp = np.empty((NY,NX))
    
    tmp[1:-1,1:-1] = (
        (u[1:-1, 2:] - u[1:-1, :-2])/dx/2 +
        (v[2:, 1:-1] - v[:-2, 1:-1])/dy/2 )
        
    return tmp

def grad():
    """
    Calcule le gradient de phi (ordre 2)
    update gradphix and gradphiy
    
    """
    global gradphix, gradphiy

    gradphix[:, 1:-1] = (phi[:, 2:] - phi[:, :-2])/dx/2
    gradphiy[1:-1, :] = (phi[2:, :] - phi[:-2, :])/dy/2

       
###
def VelocityGhostPoints(u,v):
    ### left
    u[:,  0] = u0
    v[:,  0] = 0
    ### right      
    u[:, -1] = u[:, -2] 
    v[:, -1] = v[:, -2] 
    ### bottom     
    u[0,  :] = -u[2,  :] 
    v[0,  :] = v[2,  :] 
    ### top      
    u[-1, :] = -u[-3, :] 
    v[-1, :] = v[-3, :] 
        
def PhiGhostPoints(phi):
    """
    copie les points fantomes
    tjrs Neumann

    global ==> pas de return 

    """
    ### left            
    phi[:,  0] = phi[:,  2]
    ### right             
    phi[:, -1] = -phi[:, -3]
    ### bottom   
    phi[0,  :] = phi[2,  :]
    ### top               
    phi[-1, :] = phi[-3, :]

def VelocityObstacle(ustar,vstar):
    """
    on impose une vitesse nulle sur le carré
    """
    global ObsX1, ObsY1, ObsX2, ObsY2
    ustar[ObsY1:ObsY2, ObsX1:ObsX2] = 0
    vstar[ObsY1:ObsY2, ObsX1:ObsX2] = 0


#########################################
###### MAIN: Programme principal
#########################################


###### Taille adimensionnee du domaine
### aspect_ratio = LY/LX  

aspect_ratio = float(1.)
LY = float(1.)
LX = LY/aspect_ratio

###### GRID RESOLUTION

### Taille des tableaux (points fantomes inclus)

NX = int(300)
NY = int(100)

### Position de l'obstacle
ObsX1 = 15
ObsY1 = 50
ObsX2 = ObsX1+10
ObsY2 = ObsY1+10

### Taille du domaine reel (sans les points fantomes)
nx = NX-2
ny = NY-2

###### Parametre de controle
Re = float(5e10)

###### Conditions au limites
VerticalHeatFlux = bool(0)

###### Vitesse en entrée
u0 = 10

###### PARAMÈTRE DE BOUCLE
### Nombre d'iterations
nitermax = int(10000)

### Modulo
modulo = int(50)

##### Valeurs initiales des vitesses
u = np.zeros((NY,NX))+u0
v = np.zeros((NY,NX))

####################
###### COEF POUR ADIM

### Coef du Laplacien de la vitesse
DeltaU = float(1/Re)

###### Éléments différentiels 
dx = LX/(nx-1)
dy = LY/(ny-1)

dx_2 = 1./(dx*dx)
dy_2 = 1./(dy*dy)


### ATTENTION: dt_init calculer la CFL a chaque iteration... 
dt = float(1)

t = 0. # total time


### Tableaux avec points fantomes
### Matrices dans lesquelles se trouvent les extrapolations
Resu = np.zeros((NY,NX))
Resv = np.zeros((NY,NX))

### Definition des matrices ustar et vstar
ustar = np.zeros((NY,NX))
vstar = np.zeros((NY,NX))

### Definition de divstar
divstar = np.zeros((NY,NX))

### Definition de la pression phi
phi      = np.zeros((NY,NX))
gradphix = np.zeros((NY,NX))
gradphiy = np.zeros((NY,NX))


###### CONSTRUCTION des matrices et LU décomposition

### Construcion matricielle pour l'étape de projection
LAPoisson = BuildLaPoisson() 
LUPoisson = ILUdecomposition(LAPoisson)


### Maillage pour affichage (inutile)
# ne pas compter les points fantomes
x = np.linspace(0,LX,nx) 
y = np.linspace(0,LY,ny)
[xx,yy] = np.meshgrid(x,y) 


###### CFL explicite
dt_exp = CFL_explicite()


###### Reference state
Tr = 1 - y

################
###### MAIN LOOP 
tStart = t

plt.ion()
for niter in xrange(nitermax):
    ###### Check dt
    dt_adv = CFL_advection()
    
    dt_new = min(dt_adv,dt_exp)
    
    if (dt_new < dt):
        dt = dt_new
        
    ### Avancement du temps total
    t += dt

    ###### Etape d'advection semi-Lagrangienne
    Advect()

    ###### Etape de diffusion

    ustar = Resu + dt*DeltaU*Laplacien(u) 
    vstar = Resv + dt*DeltaU*Laplacien(v) 

    ###### Conditions aux limites Vitesse
    ###### on impose sur ustar/vstar Att:ghost points
    ### left
    ustar[:,  1] = u0
    vstar[:,  1] = 0.0
    ### right
    ustar[:, -1] = ustar[:, -2]
    vstar[:, -1] = vstar[:, -2]
    ### top
    ustar[-1, :] = 0.0
    vstar[-1, :] = vstar[-2, :]
    ### bottom
    ustar[0,  :] = 0.0
    vstar[0,  :] = vstar[1, :]
        
    ###### END Conditions aux limites
    ###### Réctification de ustar et vstar pour avoir une vitesse nulle
    VelocityObstacle(ustar,vstar)
    
    ###### Etape de projection
    
    ###### Mise a jour des points fantomes pour 
    ###### calculer la divergence(ustar,vstar) 
   
    VelocityGhostPoints(ustar,vstar)

    ### Update divstar 
    divstar = divergence(ustar,vstar)


    ### Solving the linear system
    phi[1:-1,1:-1] = ResoLap(LUPoisson, RHS=divstar[1:-1,1:-1])

    ### update Pressure ghost points 

    PhiGhostPoints(phi)

    ### Update gradphi

    grad()

    u = ustar - gradphix
    v = vstar - gradphiy

    ###### Mise a jour des points fantomes

    VelocityGhostPoints(u,v)

    if (niter%modulo==0):

        ###### logfile
        sys.stdout.write(
            '\niteration: %d -- %i %%\n'
            '\n'
            'total time     = %.2e\n'
            '\n'
            %(niter,                    
              float(niter)/nitermax*100,
              t))
        
        
        ###### FIGURE draw works only if plt.ion()
        plotlabel = "t = %1.5f" %(t)
        plt.title(plotlabel)
        #plt.imshow(np.sqrt((u[1:-1,1:-1])**2 + (v[1:-1,1:-1])**2), origin="lower")
        plt.imshow(u[1:-1, 1:-1], origin="lower")
        #plt.quiver(u[::4, ::4],v[::4, ::4], units="dots", width=0.7, 
        #           scale_units="dots", scale=0.9,
        #hold=False)
        plt.axis('image')
        plt.draw()
        #plt.grid()
        #plt.plot(vstar[75,:])
        #plt.plot(Resv[75,:])
        #plt.plot(u[,:])
        

        ###### Gael's tricks interactif
        if 'qt' in plt.get_backend().lower():
            QtGui.qApp.processEvents()
plt.ioff()
