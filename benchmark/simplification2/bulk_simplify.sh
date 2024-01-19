#!/bin/bash

for filename in ~/3d_facets/facets/*; do
	echo $filename
	python3 simplify_mesh.py $filename .999 #0.5,0.3,0.2,0.1,0.05,0.02,0.01,0.005,0.001

done
