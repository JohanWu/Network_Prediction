#!/bin/bash
iter=50

rm -f *~
rm -rf Results/

for (( i=0; i<iter; i=i+1 ))
do
#echo $i
rm -f *.p
python synthetic_network_generator.py
python main.py 'linear'
python main.py 'sigmoid'
python main.py 'tanh'
python main.py 'elu'
python main.py 'selu'
python main.py 'relu'
done


