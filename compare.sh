#!/bin/sh
python2 Rayleigh_Benard.py --nx 200 --ny 100 --BFECC&
python2 Rayleigh_Benard.py --nx 200 --ny 100&
while [ `zenity --question --text "Kill python ?"` ]; do
    sleep 1
done
killall python2
