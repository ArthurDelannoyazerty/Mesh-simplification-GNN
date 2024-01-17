# Mesh Simplification 

This python 3+ mesh simplification script is a command line tool that uses Quadric Error Metrics [pdf](https://www.cs.cmu.edu/~./garland/Papers/quadrics.pdf) to iteratively contract edges that minimize the change to the overall geometry.


<p align="center">
  <img src="https://i.imgur.com/ljd5ZSu.gif" alt="Triceratops skull" width="500">
</p>

<p align="center">Mesh Simplification from original 600,000 faces to 600 faces </p>


## Supported Files

Currently, this tool supports `.obj` files without any extra steps and `.stl` files after they've been converted from binary to ASCII. The ruby script, `convertSTL.rb` was forked from [here](https://github.com/cmpolis/convertSTL) (credit to cmpolis) and can be run from the command line to convert from binary encoding to ASCII.
 
  
    $ ruby convertSTL.rb [filename of .stl to be converted]

## Usage

Once the files are in a supported format, the tool can be run after providing the 3D file and the stopping criteria as a fraction. For example if 0.3 is provided, the algorithm will stop when 70% of the faces have been deleted. A list of decreasing fractions, separated by commas, can also be provided. The command below will output three sets of data with 50%, 90%, 99.5% reductions respectively.

    $ python simplify_mesh.py /path/to/file.stl 0.5,0.1,0.005
    

## Output

The tool will output the final vertices and faces in separate files which are easy to read in via numpy. The provided script `view_mesh.py` uses pyqtgraph to view and interact with the final mesh. Run this viewer tool with the following command:

    $ python view_mesh base_file 0.5

## Details

The original algorithm contracts the edge with the minimal quadric error. For large parts, this can add significant computation time. Other implementations have instead iterated through the edges and contracted any whose error was below a cutoff value, decreasing the time required dramatically .  This version uses a hybrid of the two; first using the cutoff method to decrease the number of overall faces and then if necessary it uses the minimal quadric error. Once faces have been reduced with the cutoff method, finding the minimum error is not as demanding. Additionally, this algorithm preserves boundary edges by adding a weighted matrix to edges with only one neighboring face.

