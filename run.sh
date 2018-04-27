#!/bin/bash
iter=25

rm -f *~
rm -f *.txt

for (( i=0; i<iter; i=i+1 ))
do
#echo $i
rm -f *.p
python synthetic_network_generator.py
python main.py
done


