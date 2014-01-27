import os

ploter = "python  ./Rayleigh_Benard.py"
niter = 2000
options = " --nx 200 --ny 100 --behind --rect 15 3 -R --alpha 100 -s 5 --niter "+str(niter)+" -r 50"
var_parameter = lambda iter: " -S " +  str(10+iter) + " 0.02"
out = lambda iter: " --out sequence_" + str(iter)
verbose = " --drag_only"

simul = lambda i : ploter + options + out(i) + var_parameter(i) + verbose
pipe = ""
stdout = lambda iter : " > sequence_" + str(iter) + "_output"

for i in xrange(10):
    command = simul(i) + pipe + stdout(i)
    print "Launching command " + str(i)
    print command
    os.system(command)
