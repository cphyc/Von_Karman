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
import matplotlib.path as mpa
import matplotlib.transforms as mtr
parser = argparse.ArgumentParser(description='Von Karman streets.')
parser.add_argument('--hash', '-H', required=False, action='store_true',
                    help = 'Experimental : use a hash method for speed improvements (default : false)',
                    dest='do_hash', default=False)
parser.add_argument('--hash-size', required=False, default=30, 
                    dest='max_hash_size', nargs="?",
                    help='Control the size of the stack for the hash (default : 30)')
parser.add_argument('--BFECC', '-b', required=False, action='store_true',
                    help='Use the BFECC method for better simulation (default : false)',
                    default=False)
parser.add_argument('--tracer', '-t', required=False, type=float, default=10,
                    dest='tracers', help='Use TRACER tracers (default : 10)')
parser.add_argument('--Re', required=False, default=float(1e4),
                    dest='re',help="Reynold's number, (default : 1e4)")
parser.add_argument('--nx', required=False, type=int, default=150,
                    help="Grid size in the x direction (default : 150)")
parser.add_argument('--ny', required=False, type=int, default=80,
                    help="Grid size in the y direction (default : 80)")

parser.add_argument('--ox', required=False, type=int, default=15,
                    help="Obstacle leftest position in the x direction")

parser.add_argument('--behind', required=False, action='store_true',
                    help="Tracers behind the obstacle (default : false)")
parser.add_argument('--parallel', required=False, action='store_true',
                    help="Use parallel processing for plotting (default : false)")
parser.add_argument('--max_parallel', required=False, action='store_true',
                    help="Use parallel processing for processing (default : false)")
parser.add_argument('--nprocess', required=False, type=int, default=4,
                    help="Number of coprocessors to use (default : 4)")
parser.add_argument('--out', required=False, type=str, default="out",
                    help="Name of the file of the output (default 'out')")
parser.add_argument('--tracer-size', required=False, type=int, default=1,
                    dest="colWidth", help="Size of the tracers (default : 1)")
parser.add_argument('--assymetry', '-a', required=False, type=int, default=0,
                    dest="assym", help="Assymetry (default : None)")
parser.add_argument('--speed', '-s', required=False, type=int,
                    default=10, dest="speed", help="Speed at the left (default : 10)")
parser.add_argument('--sinus', '-S', nargs=2, required=False,
                    default=(5,10), dest="sinus", 
                    metavar=('F','A'),
                    help="Use a sinus at frequency F and amplitude A for the oscillation (default : F=5, A=10)")
parser.add_argument('--rotate', '-R', required=False,
                    default=False, action='store_true',
                    help="Use an oscillation.")
parser.add_argument('--rect', required=False, nargs=2, metavar=("WIDTH", "HEIGHT"),
                    default=(40, 40), 
                    help="The obstacle is a rectangle of size WIDTH*HEIGHT (default : 40*40).")
parser.add_argument('--circle', required=False, metavar=("RADIUS"),
                    nargs='?', 
                    help="The obstacle is a circle of radius RADIUS (default : 40) (overrides --rect option)")
parser.add_argument('--refresh', '-r',type=int,
                    required=False, default=50,
                    dest='refresh', help="Refresh rate (default : 50")
parser.add_argument('--verbose', '-v', action="store_true",
                    required=False, help="Enable verbose output (default : false)")
parser.add_argument('--movie', '-m', required=False, action="store_true",
                    help="The output is now a sequence of pictures")
parser.add_argument('--alpha', required=False, default=1000., type=float,
                    help="Absorption coefficient in the obstacle")
parser.add_argument('--niter', required=False, type=int, default=10000,
                    help="Number of iterations (def: 10.000)")

args=parser.parse_args()
freq=float(args.sinus[0])
amp=float(args.sinus[1])

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
    
# Liste des taches
if args.parallel or args.max_parallel:
    import pp
    jober = pp.Server()
    j=[]
    for i in xrange(int(args.nprocess)):
        j.append((0,lambda:0))

# Object obstacle:
class Obstacle:
    def __init__(self, xmin, ymin, w, h):
        # Boîte délimitée par xmin, xmin+w, ymin, ymin+h
        self.xmin = xmin
        self.xmax = xmin + w
        self.ymin = ymin
        self.ymax = ymin + h
        self.posX = xmin
        self.posY = ymin
        self.diag = int(numpy.sqrt(w**2+h**2))
        self.max_shape = (self.diag, self.diag)
    def __iter__(self):
        return self
    def rotation_point(self):
        # Par défaut, on tourne autour du milieu du côté gauche
        return (self.xmin, (self.ymax-self.ymin)/2)
    def next(self):
        x = self.posX
        y = self.posY
        if (self.posX + 1 < self.xmax): # Si on peut continuer sur la ligne
            self.posX+=1
        else :
            self.posY += 1 # Sinon on monte
            self.posX = self.xmin
        if x >= self.xmin and x < self.xmax and y >= self.ymin and y < self.ymax:
            return [x,y]
        else:
            self.posX = xmin
            self.posY = ymin
            raise StopIteration
class Mask:
    def __init__(self):
        self.ptList = []
    def add(self, pt):
        if not(pt in self.ptList):
            self.ptList.append(pt)
            
    # Méthode qui applique le champ factor avec sur la liste de
    # champ*vitesse
    def apply(self, ls, speed, pivot):
        px, py = pivot
        exp_fact = numpy.exp(-args.alpha*dt)
        for field,s in zip(ls, speed):
            for x,y in self.ptList:
                # On se place en référence au point de pivot pour le facteur
                sobs = numpy.sqrt((y-py)**2+(x-px)**2)*s
                # On applique la formule
                field[y,x] = sobs + (field[y,x] - sobs)*exp_fact
                         
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
    umax = max(numpy.abs(u).max(),0.01) 
    vmax = max(numpy.abs(v).max(),0.01)
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

# Calcule au (resp. av) et 1. - au
def A(field, dt, dx):
    rep = abs(field[1:-1,1:-1]) * dt / dx
    return (rep, 1. - rep)

def M(field, _) :
    M = numpy.sign(numpy.sign(field[1:-1,1:-1]) + 1.)
    return (1. - M, M)
####
def Advect(u, v, p):
    """
    Calcule la valeur interpolee qui correspond 
    a l'advection a la vitesse au temps n.
    
    travaille bien sur le domaine reel [1:-1,1:-1]

    Prend en argument u, v (vecteur vitesses), p (quantité à advecter)
    et réalise l'advection selon l'équation. 
    Retourne le champ de vecteur advecté.

    """
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
        if args.max_parallel:

            # On lance le calcul de au puis av
            ratio = dt/dx

            auJob = jober.submit(A, (u, dt, dx), modules=("numpy",))
            avJob = jober.submit(A, (v, dt, dy), modules=("numpy",))

            # On lance le calcul de Mx et My
            MxJob = jober.submit(M, (u, False), modules=("numpy",))
            MyJob = jober.submit(M, (v, False), modules=("numpy",))

            # On récupère au et 1 - au (resp. av et 1 - av)
            au, au_1 = auJob()
            av, av_1 = avJob()

            Cc = au_1 * av_1
            Ce = au * av
            Cmx = au_1 * av
            Cmy = av_1 * au

            # On récupère Mx et My
            Mx1, Mx2 = MxJob()
            My1, My2 = MyJob()

        else:
            Mx1, Mx2 = M(u, False)            
            My1, My2 = M(v, False)
            au, au_1 = A(u, dt, dx)
            av, av_1 = A(v, dt, dy)
        
            # Mx2 = numpy.sign(numpy.sign(u[1:-1,1:-1]) + 1.)
            # Mx1 = 1. - Mx2

            # # Matrice avec des 1 quand on va en haut, 
            # # 0 en bas ou au centre
            # My2 = numpy.sign(numpy.sign(v[1:-1,1:-1]) + 1.)
            # My1 = 1. - My2

            # # Matrices en valeurs absolues pour u et v
            # au = abs(u[1:-1,1:-1]) * dt/dx 
            # av = abs(v[1:-1,1:-1]) * dt/dy

            # au_1 = 1. - au
            # av_1 = 1. - av
            
            # Matrices des coefficients respectivement 
            # central, exterieur, meme x, meme y     
            Cc = (au_1) * (av_1) 
            Ce = au * av
            Cmx = (1. - au) * av
            Cmy = (1. - av) * au

        # On stocke dans la table de hashage
        if args.do_hash :
            hashtbl[h] = [Mx1, Mx2, My1, My2, au, av, Cc, Ce, Cmx, Cmy]
    # p est une liste de champs
    result = []
    for field in p:
        # on copie le tableau d'entrée l dans res
        res = field.copy()
        # Calcul des matrices de resultat: on part de l[i] et on arrive dans res[i] 
        res[1:-1,1:-1] = (Cc * field[1:-1, 1:-1] +            
                        Ce * (Mx1*My1 * field[2:, 2:] + 
                                Mx1*My2 * field[:-2, 2:] +
                                Mx2*My1 * field[2:, :-2] +
                                Mx2*My2 * field[:-2, :-2]) +  
                        Cmx * (My1 * field[2:, 1:-1] +
                               My2 * field[:-2, 1:-1]) +   
                        Cmy * (Mx1 * field[1:-1, 2:] +
                                Mx2 * field[1:-1, :-2]))
        result.append(res)
    return result

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
    # print LA0.todense()
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

    gradphix[:, 1:-1] = (phi[:, 2:] - phi[:, :-2])/dx/2
    gradphiy[1:-1, :] = (phi[2:, :] - phi[:-2, :])/dy/2

def DeltaY(t,amp,freq):
    return numpy.trunc(
        amp*numpy.sin(2*numpy.pi*freq*t))


def jacobienneH(ox1,ox2,oy):
    JH = numpy.zeros((2,2,ox2-ox1))
    JH[0,0,:]=(u[oy, ox1+1:ox2+1] - u[oy, ox1-1:ox2-1])/dx/2
    JH[1,0,:]=(u[oy+1, ox1:ox2] - u[oy-1, ox1:ox2])/dy/2
    JH[0,1,:]=(v[oy, ox1+1:ox2+1] - v[oy, ox1-1:ox2-1])/dx/2
    JH[1,1,:]=(v[oy+1, ox1:ox2] - v[oy-1, ox1:ox2])/dy/2
    
    return JH
    
def jacobienneV(oy1,oy2,ox):
    JV = numpy.zeros((2,2,oy2-oy1))
    JV[0,0,:]=(u[oy1:oy2, ox+1] - u[oy1:oy2, ox-1])/dx/2
    JV[1,0,:]=(u[oy1+1:oy2+1, ox] - u[oy1-1:oy2-1, ox])/dy/2
    JV[0,1,:]=(v[oy1:oy2, ox+1] - v[oy1:oy2, ox-1])/dx/2
    JV[1,1,:]=(v[oy1+1:oy2+1, ox] - v[oy1-1:oy2-1, ox])/dy/2
    
    return JV
    
def Drag(t):    
    """
    Calcule traînée sur l'obstacle rectangulaire en calculant la circulation de sigma sur un contour situé à 2points de l'obstacle
    """
    try:
        r = float(args.circle)
        Lcont = 2*r
        dx = 2*r
        dy = 2*r
    except:
        ds = args.rect
        dx = float(ds[0])
        dy = float(ds[1])
        Lcont = max(dx,dy)

    # On récupère l'amplitude
    _, amp = args.sinus
    amp = int(float(amp)) + 3
    
    ox = args.ox
    oy = (NY-dy)/2
    deltay = DeltaY(t,amp,freq)
    
    J1= jacobienneV(oy-amp, oy+amp+Lcont-1, ox-2) 
    J2= jacobienneH(ox-2, ox+Lcont+1, oy+amp+Lcont)
    J3= jacobienneV(oy-amp +1, oy+amp+Lcont, ox+dx+2)
    J4= jacobienneH(ox-1, ox+Lcont+2, oy-amp)
    
    #Left : on calcule sigma*ds.ex sur la gauche
    sigma1 = -dy*(-phi[oy-amp:oy+amp+Lcont-1, ox-2]
                  + 2./Re*J1[0,0,:])   #on s'arrête avant le coin en haut à gauche
    
    #top
    sigma2 = dx*(-phi[oy+amp+Lcont-1, ox-2:ox+Lcont+1]
                 + 1./Re*(J2[1,0,:]+J2[0,1,:]))  # pareil pas le troisième coin
    
    #right
    sigma3 = dy*(-phi[oy-amp+1:oy+amp+Lcont, ox+Lcont+2]
                 + 2./Re*J3[0,0,:])
    
    #bottom
    sigma4 = -dx*(-phi[oy-amp, ox-1:ox+Lcont+2]
                  + 1./Re*(J4[1,0,:]+J4[0,1,:]))
    
    return sigma1.sum() + sigma2.sum() + sigma3.sum() + sigma4.sum()
       
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
    
def VelocityObstacle(ls, t, speed):
    """
    on impose une vitesse égale à la vitesse de l'obstacle sur celui-ci
    en option, rotate qui donne le point autour duquel on tourne. La vitesse est
    alors définie comme la vitesse de rotation.
    """
    deltay = DeltaY(t, amp, freq)
    # On définit la fonction factor par deux fonctions lambdas:
    # si on ne pivote pas, le facteur de rotation vaut 1 pour tout x,y
    # si on pivote, il vaut la position relative du centre de rotation
    if args.rotate:
        # On a un point de pivot, donc notre forme est un rectangle.
        # On récupère les coordonnées de rotation
        x0, y0 = obs.rotation_point()
        # D'une part, la matrice de rotation autour du pivot d'angle A*sin(wt):
        R = mtr.Affine2D()
        R.rotate_deg_around(x0,y0,amp*numpy.sin(2*numpy.pi*freq))
        rangeX = range(0,NX)
        rangeY = range(0,NY) 
        # Pour tout point de l'obstacle, on fait son image
        mask = Mask()
        for pt in obs:
            # On arrondi
            x,y = numpy.floor(R.transform_point(pt))
            # x,y = int(img[0]), int(img[1])
            mask.add([x,y])
        # On parcourt maintenant le masque
        mask.apply(ls, speed, obs.max_shape)

    else:
        # On calcule le facteur de pénalisation
        exp_fact = numpy.exp(-args.alpha*dt)
    
        try: # On a un cercle
            r = int(args.circle) # échoue si vaut None
            ox = args.ox + r
            oy = NY/2 + deltay + args.assym
            # Bornes de x
            for x in xrange(-r,r+1):
                ym = int(numpy.sqrt(r**2 - x**2))
                xabs = x+ox
                # Bornes de y (pour le cercle)
                for y in xrange(-ym, ym+1):
                    yabs = y+oy
                    for el,sobs in zip(ls,speed):
                        # On calcule la vitesse en ce point
                        el[yabs, xabs] = sobs+(el[yabs,xabs]-sobs)*exp_fact
        except TypeError : # Si on a un rectangle
            ds = args.rect
            dx=float(ds[0])
            dy=float(ds[1])+args.assym
            # On se place au centre + offset
            y1 = (NY-dy)/2+args.assym + deltay
            y2 = y1 + dy
            x1 = args.ox
            x2 = x1 + dx
            # Tableau de la taille du carré. La coordonné (x,y) a comme valeur:
            # factor(x,y) (càd la position si on tourne, sinon 1)
            farr = numpy.ones((dx, dy))
            for el,sobs in zip(ls,speed):
                # Le tableaux des vitesses est le le tableaux des positions * vitesse
                el[y1:y2, x1:x2]=sobs+(el[y1:y2, x1:x2]-sobs)*exp_fact
            
def ploter(param, drags, times):
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
    if args.behind:
        vmi = 0.3
    else:
        vmi = 0
    plt.imshow(T[1:-1, 1:-1], vmin=vmi, vmax=1,figure=0)
    # plt.ioff()
    # plt.plot(T[50,1:-1],hold=False)
    #plt.quiver(u[::4, ::4],v[::4, ::4], units="dots", width=0.7, 
    #           scale_units="dots", scale=0.9,
    #hold=False)
    x = args.nx
    y = args.ny
    plt.axis([0,x,0,y])
    # plt.draw()
    plt.grid()
    #plt.plot(vstar[75,:])
    #plt.plot(Resv[75,:])
    #plt.plot(u[,:])
    # On sauvegarde dans une liste drags et times
    # out_name = args.out + ".png"
    if args.movie:        
        if args.verbose:
            print "Saving image number " + str(param['niter'])
            out_name = args.out + "_" + str(param['niter']) + ".png"
    else:
        out_name = args.out + ".png"
    plt.savefig(out_name)
    plt.clf()
    plt.plot(times[1:],drags[1:])
    plt.axis('auto')
    plt.savefig("drag.png")

    ###### Gael's tricks interactif
    # if 'qt' in plt.get_backend().lower():
    #     QtGui.qApp.processEvents()
                
#########################################
###### MAIN: Programme principal
#########################################


###### Taille adimensionnee du domaine
### aspect_ratio = LY/LX  

aspect_ratio = float(args.ny*1.0/args.nx)
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
nitermax = int(args.niter)

##### Valeurs initiales des vitesses
u = numpy.zeros((NY,NX))+u0
v = numpy.zeros((NY,NX))
if use_tracer > 0:
    if args.behind:
        T = numpy.ones((NY,NX))
    else:
        T = numpy.zeros((NY,NX))
        
##### INITIALISATION DE L'OBSTACLE
xmin = args.ox
ymin = (NY - int(args.rect[1]))/2
obs = Obstacle(int(xmin), int(ymin), int(args.rect[0]), int(args.rect[1]))

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

# On initialise le tableau des temps et des drags
drags = []
times = []

for niter in xrange(nitermax):
    ###### Check dt
    dt_adv = CFL_advection()
    dt_new = min(dt_adv,dt_exp)
    
    if (dt_new < dt):
        dt = dt_new
    ### Avancement du temps total
    t += dt
    
    ### Calcul des vitesses de l'obstacle
    uobs = 0
    vobs = numpy.pi*2*freq*amp*numpy.cos(numpy.pi*2*freq*t)

    ###### Etape d'advection semi-lagrangienne utilisant la méthode BFECC
    def lets_advect(p, BFECC, speeds):
        if BFECC :
            p3 = Advect(u, v, p)          
            p2 = Advect(-u, -v, p3)
            prect = p +1./4*numpy.subtract(p,p2)
            p1 = Advect(u, v, prect)
            VelocityObstacle(p1, t, speeds)
            return p1
        else :
            p = Advect(u, v, p)
            VelocityObstacle(p,t, speeds)
            return p
        
    # param = {'args':args, 'u':u, 'v':v, 't':t, 'dt':dt,
    #          'dy':dy, 'dx':dx, 'freq':freq, 'amp':amp, 'NX':NX, 'NY':NY
    #          }
    # if args.max_parallel:
    #     ResuP = jober.submit(lets_advect, ([u], args.BFECC, param, [uobs]),
    #                           (Advect,VelocityObstacle, DeltaY),("numpy",))
    #     ResvP = jober.submit(lets_advect, ([v], args.BFECC, param, [vobs]),
    #                           (Advect,VelocityObstacle, DeltaY),("numpy",))
    #     TP = jober.submit(lets_advect, ([T], args.BFECC, param, [0]),
    #                        (Advect,VelocityObstacle, DeltaY),("numpy",))
    #     [Resu] = ResuP()
    #     [Resv] = ResvP()
    #     [T] = TP()
    # else:
    Resu,Resv,T = lets_advect([u,v,T], args.BFECC, [uobs,vobs,0])


    ###### Etape de diffusion
    ustar = Resu + dt*DeltaU*Laplacien(u) 
    vstar = Resv + dt*DeltaU*Laplacien(v) 

    ###### Conditions aux limites Vitesse
    ###### on impose sur ustar/vstar Att:ghost points

    ###### Réctification de ustar et vstar pour avoir une vitesse nulle
    VelocityObstacle([ustar,vstar], t, [uobs,vobs])
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
    
    #### Calcul de la traînée

    
    ###### Mise a jour des points fantomes
    VelocityGhostPoints(u,v)
    if use_tracer :
        TraceurGhostPoint(T)
    # On met à jour les drags et le temps pour faire le suivi
    drag = Drag(t)
    drags += [drag]
    times += [t]
    
    if (niter%args.refresh==0):
        if args.verbose :
            
            ###### logfile
            sys.stdout.write(
                '\niteration: %d -- %i %%\n'
                '\n'
                'total time     = %.2e\n'
                '\n'
                'drag           = %.2e\n'
                '\n'
                %(niter,
                  float(niter)/nitermax*100,
                  t,drag))
                  
        

        param={"args":args, "t":t, "T":T, "nitermax" : nitermax,
               "dx":dx, "dy":dy, "niter":niter, "dt":dt}
        if args.parallel or args.max_parallel:
            j.append((niter,
                    jober.submit(ploter,(param, drags, times),(DeltaY,)
                                 ,("matplotlib.pyplot",) )))
            # On récupère et supprime le 1er él,
            # et on attend la fin de son exécution
            niter0, wait_next = j.pop(0)
            if args.verbose:
                print "We're at ",niter,". Waiting for the end of :", niter0
            wait_next()
        else:
            ploter(param, drags, times)

if args.parallel or args.max_parallel:
    print "Waiting the end of the threads"
    [w() for n,w in j]
