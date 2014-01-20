#!/usr/bin/python2
# -*- coding:utf-8 -*-
#
# Convection 2D schema explicite
#      avec points fantomes
#
import sys
import numpy
import scipy.sparse as sp
import scipy.sparse.linalg as lg
import argparse

parser = argparse.ArgumentParser(description='Von Karman streets.')
parser.add_argument('--hash', '-H', required=False, action='store_true',
                    help = 'Use a hash method for speed improvements',
                    dest='do_hash', default=False)
parser.add_argument('--hash-size', required=False, default=30, 
                    dest='max_hash_size', nargs="?",
                    help='Control the size of the stack for the hash')
parser.add_argument('--BFECC', '-b', required=False, action='store_true',
                    help='Use the BFECC method for better simulation',
                    default=False)
parser.add_argument('--tracer', '-t', required=False, type=float, default=10,
                    dest='tracers', help='Use TRACER tracers, default:10')
parser.add_argument('--Re', required=False, default=float(1e15),
                    dest='re',help="Reynold's number, default:1e15")
parser.add_argument('--nx', required=False, type=int, default=300,
                    help="Grid size in the x direction")
parser.add_argument('--ny', required=False, type=int, default=100,
                    help="Grid size in the y direction")
parser.add_argument('--ox', required=False, type=int, default=15,
                    help="Obstacle leftest position in the x direction")
parser.add_argument('--behind', required=False, action='store_true',
                    help="Tracers behind the obstacle")
parser.add_argument('--parallel', required=False, action='store_true',
                    help="Use parallel processing for plotting")
parser.add_argument('--max_parallel', required=False, action='store_true',
                    help="Use parallel processing for processing")
parser.add_argument('--nprocess', required=False, type=int, default=2,
                    help="Number of coprocessors to use")
parser.add_argument('--out', required=False, default="test.png")
parser.add_argument('--tracer-size', required=False, type=int, default=1,
                    dest="colWidth", help="Size of the tracers")
parser.add_argument('--assymetry', '-a', required=False, type=int, default=0,
                    dest="assym", help="Assymetry")
parser.add_argument('--speed', '-s', required=False, type=int,
                    default=1, dest="speed", help="Speed at the left")
parser.add_argument('--sinus', '-S', nargs=2, required=False,
                    default=(0,0), dest="sinus", 
                    metavar=('F','A'),
                    help="Use a sinus at frequency F and amplitude A.")
parser.add_argument('--rect', required=False, nargs=2, metavar=("WIDTH", "HEIGHT"),
                    default=(40, 40), 
                    help="The obstacle is a rectangle of size WIDTH*HEIGHT (default : 40*40).")
parser.add_argument('--circle', required=False, metavar=("RADIUS"),
                    type=int, 
                    help="The obstacle is a circle of radius RADIUS (default : 40) (overrides --rect option).")
parser.add_argument('--refresh', '-r',type=int,
                    required=False, default=50,
                    dest='refresh', help="Refresh rate")
parser.add_argument('--verbose', '-v', action="store_true",
                    required=False, 
                    help="Enable output")

args=parser.parse_args()
freq=int(args.sinus[0])
amp=int(args.sinus[1])

if args.do_hash : print 'Using hash algorithm...'
if args.BFECC : print 'Using BFECC method...'
else: print 'No BFECC...'
if args.tracers > 0 : print 'Using', args.tracers, 'tracers...'
print "Refresh rate :", args.refresh
print "Reynold's number :", args.re

if args.tracers > 0:
    use_tracer=True
else:
    use_tracer=False

# Petite table de hashage pour optimiser
if args.do_hash:
    hashtbl = dict()
    h = 0

###### affichage graphique
# import matplotlib.pyplot as plt
# if 'qt' in plt.get_backend().lower():
#     try:
#         from PyQt4 import QtGui
#     except ImportError:
#         from PySide import QtGui

                 
def CFL_advection():
    """
    Condition CFL Advection pour dt
    return le nouveau 'dt' tel que

    abs(V) * dt < dx

    dt_new = 0.8*min(dx/umax,dy/vmax)

    """
    precautionADV = 0.8
    umax = numpy.amax(numpy.abs(u))
    vmax = numpy.amax(numpy.abs(v))
    divider = max(umax/dx, vmax/dy, 0.01/dx, 0.01/dy)
    dt_cfa = precautionADV / divider
    # umax = max(numpy.abs(u).max(),0.01) 
    # vmax = max(numpy.abs(v).max(),0.01)
    # dt_cfa = precautionADV * min(dx/umax,dy/vmax)
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
def Advect(u, v, p, param):
    """
    Calcule la valeur interpolee qui correspond 
    a l'advection a la vitesse au temps n.
    
    travaille bien sur le domaine reel [1:-1,1:-1]

    Prend en argument u, v (vecteur vitesses), p (quantités à advecter)
    et réalise l'advection selon l'équation. 
    Retourne le champ de vecteur advecté.

    """
    args=param['args']
    dx=param["dx"]
    dy=param['dy']
    dt=param['dt']
    u,v = param['u'], param['v']
    def check_hash () :
        """ Retourne un booléen si h est dans la hashtbl """
        global hashtbl, h
        h = hash(u.tostring() + v.tostring())
        if h in hashtbl:
            return True
        else:
            new_hashtbl = dict()
            try: 
                i = 0
                while True:
                    i+=1
                    if i > args.max_hash_size :
                        raise KeyError
                    k, val = hashtbl.popitem()
                    new_hashtbl[k] = val
            except KeyError:
                hashtbl = new_hashtbl.copy()

    if args.do_hash and check_hash():
        # On utilise une table de hashage pour ne pas recalculer tout à chaque fois
        # c'est un dictionnaire qui associe au hash de u, v les valeurs utiles pour 
        # l'advection 1/2 lagrangienne
        Mx1, Mx2, My1, My2, au, av, Cc, Ce, Cmx, Cmy = hashtbl[h]
    else:
        # On n'a pas encore calculé, on le fait donc
        # Matrice avec des 1 quand on va a droite, 
        # 0 a gauche ou au centre
        Mx2 = numpy.sign(numpy.sign(u[1:-1,1:-1]) + 1.)
        Mx1 = 1. - Mx2

        # Matrice avec des 1 quand on va en haut, 
        # 0 en bas ou au centre
        My2 = numpy.sign(numpy.sign(v[1:-1,1:-1]) + 1.)
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

        # On stocke dans la table de hashage
        if args.do_hash :
            hashtbl[h] = [Mx1, Mx2, My1, My2, au, av, Cc, Ce, Cmx, Cmy]

    # on copie le tableau d'entrée l dans res
    # res = p.copy()
    # Calcul des matrices de resultat : on part de l[i] et on arrive dans res[i] 
    return [(Cc * p0[1:-1, 1:-1] +            
                      Ce * (Mx1*My1 * p0[2:, 2:] + 
                            Mx1*My2 * p0[:-2, 2:] +
                            Mx2*My1 * p0[2:, :-2] +
                            Mx2*My2 * p0[:-2, :-2]) +  
                      Cmx * (My1 * p0[2:, 1:-1] +
                             My2 * p0[:-2, 1:-1]) +   
                      Cmy * (Mx1 * p0[1:-1, 2:] +
                             Mx2 * p0[1:-1, :-2])) for p0 in p]
    return res
    # def compute (p0):
    #     return (Cc * p0[1:-1, 1:-1] +
    #             Ce * (  Mx1*My1 * p0[2:, 2:] + 
    #                     Mx1*My2 * p0[:-2, 2:] +
    #                     Mx2*My1 * p0[2:, :-2] +
    #                     Mx2*My2 * p0[:-2, :-2]) +  
    #             Cmx * ( My1 * p0[2:, 1:-1] +
    #                     My2 * p0[:-2, 1:-1]) +   
    #                     Cmy * (Mx1 * p0[1:-1, 2:] +
    #                     Mx2 * p0[1:-1, :-2]))
    # return map(compute, p)

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
    dataNXi = [numpy.ones(NXi), -2*numpy.ones(NXi), numpy.ones(NXi)]   
    
    ### Conditions aux limites : Neumann à gauche, rien à droite
    dataNXi[2][1]     = 2.  # SF left
    # dataNXi[0][NXi-2] = 2.  # SF right

    ###### AXE Y
    ### Diagonal terms
    dataNYi = [numpy.ones(NYi), -2*numpy.ones(NYi), numpy.ones(NYi)] 
   
    ### Conditions aux limites : Neumann 
    dataNYi[2][1]     = 2.  # SF low
    dataNYi[0][NYi-2] = 2.  # SF top

    ###### Their positions
    offsets = numpy.array([-1,0,1])                    
    DXX = sp.dia_matrix((dataNXi,offsets), shape=(NXi,NXi)) * dx_2
    DYY = sp.dia_matrix((dataNYi,offsets), shape=(NYi,NYi)) * dy_2
    #print DXX.todense()
    #print DYY.todense()
    
    ####### 2D Laplace operator
    LAP = sp.kron(sp.eye(NYi,NYi), DXX) + sp.kron(DYY, sp.eye(NXi,NXi))
    
    ####### BUILD CORRECTION MATRIX

    ### Upper Diagonal terms
    dataNYNXi = [numpy.zeros(NYi*NXi)]
    offset = numpy.array([1])

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
    rst = numpy.empty((NY,NX))

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
    tmp = numpy.empty((NY,NX))
    
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

    gradphix[:, 1:-1] = (phi[:, 2:] - phi[:, :-2])/(dx*2)
    gradphiy[1:-1, :] = (phi[2:, :] - phi[:-2, :])/(dy*2)

       
###
def VelocityGhostPoints(u,v):
    ### left
    u[:,  0] = args.speed
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

def TraceurGhostPoint(T):
    if args.behind:
        T[:, 1] = 1
    else:
        for i in xrange(0, args.colWidth):
            T[i::DeltaTraceur, 0] = 1

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

def VelocityObstacle(ls ,t, param):
    """
    on impose une vitesse nulle sur le carré
    """
    args=param['args']
    freq=param['freq']
    amp=param['amp']
    dx, dy = param["dx"], param["dy"]
    NX, NY = param["NX"], param["NY"]
    deltay = numpy.trunc(
        amp*numpy.sin(numpy.pi*2*freq*t))
    try: # On a un cercle
        r = int(args.circle) # échoue si vaut None
        ox = args.ox + r
        oy = NY/2 + deltay
        for x in xrange(-r,r+1):
            ym = int(numpy.sqrt(r**2 - x**2))
            xabs = x+ox
            for y in xrange(-ym, ym+1):
                yabs = y+oy
                for el in ls:
                    el[yabs, xabs] = 0
    except TypeError : # Si on a un rectangle
        # On crée le chemin
        dx,dy = args.rect
        y1 = (NY-dy)/2+args.assym + deltay
        y2 = y1 + dy
        x1 = args.ox
        x2 = x1 + dx
        # On se place au centre + offset
        for el in ls:
            el[y1:y2, x1:x2] = 0
            
def ploter(param):
    import matplotlib.pyplot as plt
    # plt.ion()
    args = param['args']
    t = param['t']
    T = param['T']

    ## FIGURE draw works only if plt.ion()
    plotlabel = "t = %1.5f" %(t)
    plt.title(plotlabel)
    #plt.imshow(numpy.sqrt((u[1:-1,1:-1])**2 + (v[1:-1,1:-1])**2), origin="lower")
    #plt.imshow(u[1:-1, 1:-1], origin="lower")
    plt.imshow(T[1:-1, 1:-1], vmin=0, vmax=1,figure=0)
    # plt.ioff()
    # plt.plot(T[50,1:-1],hold=False)
    #plt.quiver(u[::4, ::4],v[::4, ::4], units="dots", width=0.7, 
    #           scale_units="dots", scale=0.9,
    #hold=False)
    plt.axis('image')
    # plt.draw()
    plt.grid()
    #plt.plot(vstar[75,:])
    #plt.plot(Resv[75,:])
    #plt.plot(u[,:])
    plt.savefig(args.out)

    ###### Gael's tricks interactif
    # if 'qt' in plt.get_backend().lower():
    #     QtGui.qApp.processEvents()
                
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
NX = args.nx
NY = args.ny

### Écart entre les traceurs
DeltaTraceur = NY/args.tracers

### Taille du domaine reel (sans les points fantomes)
nx = NX - 2
ny = NY - 2

###### Parametre de controle
Re = float(args.re)

###### Vitesse en entrée
u0 = 10

###### PARAMÈTRE DE BOUCLE
### Nombre d'iterations
nitermax = int(80)

##### Valeurs initiales des vitesses
u = numpy.zeros((NY,NX))+u0
v = numpy.zeros((NY,NX))
if use_tracer > 0:
    T = numpy.zeros((NY,NX))

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
Resu = numpy.zeros((NY,NX))
Resv = numpy.zeros((NY,NX))

### Definition des matrices ustar et vstar
ustar = numpy.zeros((NY,NX))
vstar = numpy.zeros((NY,NX))

### Definition de divstar
divstar = numpy.zeros((NY,NX))

### Definition de la pression phi
phi      = numpy.zeros((NY,NX))
gradphix = numpy.zeros((NY,NX))
gradphiy = numpy.zeros((NY,NX))


###### CONSTRUCTION des matrices et LU décomposition

### Construcion matricielle pour l'étape de projection
LAPoisson = BuildLaPoisson() 
LUPoisson = ILUdecomposition(LAPoisson)

### Maillage pour affichage (inutile)
# ne pas compter les points fantomes
x = numpy.linspace(0,LX,nx) 
y = numpy.linspace(0,LY,ny)

###### CFL explicite
dt_exp = CFL_explicite()


################
###### MAIN LOOP 

# Liste des taches. Par défaut, on met 4 fonctions qui renvoient 0
if args.parallel or args.max_parallel:
    import pp
    jober = pp.Server()
    j=[]
    for i in xrange(int(args.nprocess)):
        j.append((0,lambda:0))
        
for niter in xrange(nitermax):
    ###### Check dt
    dt_adv = CFL_advection()
    dt_new = min(dt_adv,dt_exp)
    
    if (dt_new < dt):
        dt = dt_new
    ### Avancement du temps total
    t += dt

    ###### Etape d'advection semi-lagrangienne utilisant la méthode BFECC
    def lets_advect(p, BFECC, param):
        t=param['t']
        u,v=param['u'], param['v']
        if BFECC :
            p3 = Advect(u, v, p, param)          
            p2 = Advect(-u, -v, p3, param)
            p1 = Advect(u, v, p + 1./3*(p - p2), param)
            VelocityObstacle(p1,t,param)
            return p1
        else :
            p1 = Advect(u, v, p, param)
            VelocityObstacle(p1,t, param)
            return p1
        
    param = {'args':args, 'u':u, 'v':v, 't':t, 'dt':dt,
             'dy':dy, 'dx':dx, 'freq':freq, 'amp':amp, 'NX':NX, 'NY':NY}
    if args.max_parallel:
        ResuP = jober.submit(lets_advect, ([u], args.BFECC, param), (Advect,VelocityObstacle),("numpy",))
        ResvP = jober.submit(lets_advect, ([v], args.BFECC, param), (Advect,VelocityObstacle),("numpy",))
        TP = jober.submit(lets_advect, ([T], args.BFECC, param), (Advect,VelocityObstacle),("numpy",))
        Resu[1:-1, 1:-1] = ResuP()
        Resv[1:-1, 1:-1] = ResvP()
        T[1:-1, 1:-1] = TP()
    else:
        Resu[1:-1, 1:-1],Resv[1:-1, 1:-1],T[1:-1, 1:-1] = lets_advect([u,v,T],
                                                                       args.BFECC,
                                                                        param)

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
    ###### Réctification de ustar et vstar pour avoir une vitesse nulle
    VelocityObstacle([ustar,vstar],t, param)
    ###### END Conditions aux limites

    ###### Etape de projection
    
    ###### Mise a jour des points fantomes pour 
    ###### calculer la divergence(ustar,vstar) 
    VelocityGhostPoints(ustar,vstar)

    ### Update divstar 
    divstar = divergence(ustar,vstar)

    ### Solving the linear system
    phi[1:-1,1:-1] = ResoLap(LUPoisson, RHS=divstar[1:-1,1:-1])

    ### Update tracer ghost points 
    PhiGhostPoints(phi)

    ### Update gradphi
    grad()

    u = ustar - gradphix
    v = vstar - gradphiy

    ###### Mise a jour des points fantomes
    VelocityGhostPoints(u,v)
    if use_tracer :
        TraceurGhostPoint(T)

    if (niter%args.refresh==0):
        if args.verbose :
            ###### logfile
            sys.stdout.write(
                '\niteration: %d -- %i %%\n'
                '\n'
                'total time     = %.2e\n'
                '\n'
                %(niter,
                  float(niter)/nitermax*100,
                  t))

        param={"args":args, "t":t, "T":T, "nitermax" : nitermax,
               "dx":dx, "dy":dy}
        if args.parallel or args.max_parallel:
            j.append((niter,
                    jober.submit(ploter,(param,),(),("matplotlib.pyplot",) )))
            # On récupère et supprime le 1er él,
            # et on attend la fin de son exécution
            niter0, wait_next = j.pop(0)
            print "We're at ",niter,". Waiting for the end of :", niter0
            wait_next()
        else:
            ploter(param)

if args.parallel or args.max_parallel:
    print "Waitng the end of the threads"
    [w() for n,w in j]          
