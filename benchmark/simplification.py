import os
import time
import numpy as np
from util.mesh import Mesh
from benchmark import performance_metrics

from stl import mesh
import numpy as np
import csv

def convert_stl_to_obj(stl_file_path, folder_obj, filename):
    obj_file_path = (folder_obj + filename).replace('.stl', '.obj')
    stl_mesh = mesh.Mesh.from_file(stl_file_path)
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
    mesh = Mesh(filepath)

    target_v = int(len(mesh.vs) * simplification_rate)  # Use the rate in the loop

    if target_v >= mesh.vs.shape[0]:
        print(f"[ERROR]: Target vertex number for rate {simplification_rate} should be smaller than {mesh.vs.shape[0]}!")
        return None

    simp_mesh = mesh.simplification(target_v)
    return simp_mesh


def simplification2(filepath, simplification_rate):
    print("simplification 2 : ", filepath, simplification_rate)

def simplification_gnn(filepath, simplification_rate):
    print("simplification GNN : ", filepath, simplification_rate)
    


def main():
    

    simplification_functions = [simplification1, simplification2, simplification_gnn]
    file_formats             = ["obj"          ,"stl"           , "stl"             ]

    base_input_data_folder = "benchmark\data\input\\"
    base_output_data_folder= "benchmark\data\output\mesh\\"
    
    
    # for filename in os.listdir(base_input_data_folder+"stl"):                                 # decomment to transform every stl to obj                         
    #     f = os.path.join(base_input_data_folder+"stl", filename)
    #     if os.path.isfile(f):
    #         convert_stl_to_obj(f, base_input_data_folder+"obj\\", filename)
    

    for index_function, simplification_function in enumerate(simplification_functions):         # for every simplification
        data_folder = base_input_data_folder + file_formats[index_function]
        for filename in os.listdir(data_folder):                                                # for every good data file
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

                    metrics = performance_metrics(Mesh(f), simplified_mesh, f"data/output/metrics_rate_{simplification_rate:.1f}.csv", simplification_rate)

                    data_to_save = [f, simplification_rate, index_function, execution_time, metrics['hausdorff_distance'], metrics['chamfer_distance'], metrics['curvature_error'], metrics['memory_usage']]

                    with open("benchmark/data/output/csv/measures.csv", mode='a', newline='') as file_csv:
                        csv.writer(file_csv).writerow(data_to_save)




if __name__ == "__main__":
    main()

