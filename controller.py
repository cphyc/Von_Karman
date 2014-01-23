import os

ploter = "python2 ../Rayleigh_Benard.py"
niter = 2000
options = " --nx 200 --ny 100 --behind --rect 15 3 -R --alpha 100 -s 5 --parallel --niter "+str(niter)+" -r 50"
var_parameter = lambda iter: " -S 10 " + str(0.001*iter)
out = lambda iter: " --out sequence_" + str(iter)
verbose = " --drag_only"
simul = lambda i : ploter + options + out(i) + var_parameter(i) + verbose
pipe = " | tail -n +5 | head -n -1"
stdout = lambda iter : " > sequence_" + str(iter) + "_output"

for i in xrange(10,30,3):
    command = simul(i) + pipe + stdout(i)
    print "Launching command " + str(i)
    print command
    os.system(command)
