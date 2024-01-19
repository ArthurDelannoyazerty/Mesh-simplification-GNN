
import numpy as np
import os
import pandas as pd
import shutil

import time
from util.mesh import Mesh as Mesh2
from benchmark import performance_metrics
from plotting import plot_graphs 
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
import stl
import csv
from plotting import plot_graphs, load_data
from plot_table import plot_table , group_and_sort
from simplification2.simplify_meshh import *


def convert_stl_to_obj(stl_file_path, folder_obj, filename):
    obj_file_path = (folder_obj + filename).replace('.stl', '.obj')
    stl_mesh = stl.mesh.Mesh.from_file(stl_file_path)
    vertices = stl_mesh.vectors.reshape(-1, 3)
    unique_vertices, faces = np.unique(vertices, axis=0, return_inverse=True)
    faces = faces.reshape(-1, 3) + 1  # OBJ files are 1-indexed
    with open(obj_file_path, 'w') as obj_file:
        for vertex in unique_vertices:
            obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            obj_file.write(f"f {face[0]} {face[1]} {face[2]}\n")
    return obj_file_path




def simplification1(filepath, simplification_rate):
    print("simplification 1 : ", filepath, simplification_rate)
    mesh = Mesh2(filepath)

    target_v = int(len(mesh.vs) * simplification_rate) 

    if target_v >= mesh.vs.shape[0]:
         print(f"[ERROR]: Target vertex number for rate {simplification_rate} should be smaller than {mesh.vs.shape[0]}!")
         return None

    simp_mesh = mesh.simplification(target_v)
    print(simp_mesh)

    return simp_mesh 



def simplificationQm(filepath, simplification_rate):
    print("simplification 2 : ", filepath, simplification_rate)

    
        # Call the simplification2 function and get the simplified mesh
    simp_mesh, = simplification2(filepath, simplification_rate)
    print('copy begin')

    print("test", simp_mesh)
    return simp_mesh

   






def simplification_gnn(filepath, simplification_rate):
    print("simplification GNN : ", filepath, simplification_rate)
    


def main():
   

    simplification_functions = [simplification1, simplificationQm, simplification_gnn]
    file_formats             = ["obj"          ,"stl"           , "stl"             ]

    base_input_data_folder = "benchmark\data\input\\"
    base_output_data_folder= "benchmark\data\output\mesh\\"

    csv_filepath = "benchmark/data/output/csv/measures.csv"
    os.remove(csv_filepath)

   
    with open(csv_filepath, mode='a', newline='') as file_csv:
        csv.writer(file_csv).writerow(["filepath", "simplification_rate", "index_simplification_method", "time", "hausdorff_distance", "chamfer_distance", "curvature_error", "memory"])
    
    
    # for filename in os.listdir(base_input_data_folder+"stl"):                                 # decomment to transform every stl to obj                         
    #     f = os.path.join(base_input_data_folder+"stl", filename)
    #     if os.path.isfile(f):
    #         convert_stl_to_obj(f, base_input_data_folder+"obj\\", filename)
    

    for index_function, simplification_function in enumerate(simplification_functions):
        data_folder = os.path.join(base_input_data_folder, file_formats[index_function])
        for filename in os.listdir(data_folder):
            f = os.path.join(data_folder, filename)

            if os.path.isfile(f):
                for simplification_rate in np.arange(0.2, 1.0, 0.2):                            # for every simplification rate
                    start_time = time.time()
                    
                    try:
                        simplified_mesh = simplification_function(f, simplification_rate)
                    except:
                        continue
                    end_time = time.time()
                    execution_time = end_time - start_time 

                    if simplified_mesh==None: continue
                    
                    output_filepath = (base_output_data_folder + 
                                       file_formats[index_function] + 
                                       "\\" + 
                                       filename[:-4] +                                          # Name of file
                                       "_" + 
                                       str(index_function) +                                    # number of simplification function
                                       "_" + 
                                       str(simplification_rate*10_000).split(".")[0] +          # simplification rate * 10 000
                                       "." + 
                                       file_formats[index_function])                            # File format
                    simplified_mesh.save(output_filepath)

                    metrics = performance_metrics(Mesh2(f), simplified_mesh, f"data/output/metrics_rate_{simplification_rate:.1f}.csv", simplification_rate)
                    metricss = performance_metrics(f, simplified_mesh, csv_filepath, simplification_rate)
                    data = [f, simplification_rate, index_function, execution_time,
                                    metricss['hausdorff_distance'], metricss['chamfer_distance'],
                                    metricss['curvature_error'], metricss['memory_usage']]

                    print(data)
                    
                    with open(csv_filepath, mode='a', newline='') as file_csv:
                        csv.writer(file_csv).writerow(data)

                    data_to_save = [f, simplification_rate, index_function, execution_time, metrics['hausdorff_distance'], metrics['chamfer_distance'], metrics['curvature_error'], metrics['memory_usage']]
                    print(data_to_save)

                    with open(csv_filepath, mode='a', newline='') as file_csv:
                        csv.writer(file_csv).writerow(data_to_save)


 





if __name__ == "__main__":
    main()


    file_path = 'benchmark/data/output/csv/measures.csv'
    data = load_data(file_path)
    
    # You want to group by 'simplification_rate' and then sort by 'Time' within each group
    grouped_data = group_and_sort(data, 'simplification_rate', 'simplification_rate')
    
    # Now, 'highlight_columns' should be the columns you want to highlight, not the grouping column
    highlight_cols = ['CD', 'simplification_rate']  # Replace with actual columns you want to highlight
    
    plot_table(grouped_data, highlight_columns=highlight_cols)
    df = load_data('benchmark\data\output\csv\measures.csv')
    dv = load_data('benchmark\data\output\csv\metrics_results.csv')
    plot_graphs(df)
    plot_graphs(dv)
    
  

    
