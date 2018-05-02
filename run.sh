#!/bin/bash
iter=10

rm -f *~
rm -f ./Results/*.txt

for (( i=0; i<iter; i=i+1 ))
do
#echo $i
rm -f *.p
python synthetic_network_generator.py
python main.py
done


