#!/bin/bash
iter=50

rm -f *~
rm -rf Results/

for (( i=0; i<iter; i=i+1 ))
do
#echo $i
rm -f *.p
python synthetic_network_generator.py
python main.py
done


