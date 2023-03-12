#!/bin/bash
gcc -Wall -ansi final1.c -o a.out -lm -O3
rm -r outputs
mkdir outputs

# Part (a)
./a.out 2 outputs/data1.csv
python3 final.py -s -c -d outputs/data1.csv

# Part (b)
./a.out 10 outputs/data2.csv
rm a.out
python3 final.py -s -c -d outputs/data2.csv
