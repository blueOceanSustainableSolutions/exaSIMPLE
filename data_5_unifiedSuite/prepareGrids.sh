#!/bin/bash

# Prepare the grids
#rm -f gridStats.csv
#touch gridStats.csv
echo "grid,nCells" >> gridStats.csv

for grid in grids/*.cgns; do
#    filename=$(basename "$grid" | cut -d. -f1)
    filename=$(basename -- "$grid")
    filename="${filename%.*}"
    cgnsutil -g $grid -o ./grids_refresco/${filename}_refresco.cgns > /dev/null
    nCells=$(grep nTotalCells ./grids_refresco/${filename}_refresco_quality_domain1.xml | sed "s/\s*<nTotalCells>//g" | sed "s/<.*//g")
    echo $filename","$nCells >> gridStats.csv
    echo "  " $filename $nCells
done
