#!/bin/bash

# Purpose: convert lines of file ($1) into one line and offset list

text=""
offsetList=""
offset=0
IFS=""

while read -r line
do
	text+="$line"
	offsetList+="$offset, "
	offset=$(echo "$(echo "$line" | wc -c) - 1 + $offset" | bc -l)
done < "$1"
offsetList+="$offset"

echo $text
echo $offsetList
