#!/bin/bash
iter=100

for (( i=0; i<100; i=i+1 ))
do
#echo $i
rm -f *.p
python synthetic_network_generator.py
python main.py
done


