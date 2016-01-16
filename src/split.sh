#!/bin/bash

# Purpose: split lines of file ($1) at "_" and write the parts to different files
# e. g.: "This is _a_b_." into:
# static.txt:
# This is 
# .
# ab.txt:
# a
# b

sed 's/_/\n/g' $1 > output.txt
sed -n '1~3p' output.txt > static.txt
sed -n '2~3p' output.txt > a.txt
sed -n '0~3p' output.txt > b.txt
paste a.txt b.txt | tr '\t' '\n' > ab.txt
