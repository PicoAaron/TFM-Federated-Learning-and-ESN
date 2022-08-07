import json
import os
import numpy as np


if __name__ == "__main__":

    path = './results/experiment_results'
    path_processed = './results/processed_results'
    files = os.listdir(path)

    # Read the results of the experiment
    #----------------------------------------------------------
    data = []

    for file_name in files:
        
        with open(f'{path}/{file_name}') as file:
            #data = json.load(file)
            data.append(json.load(file))

    #print(data)
    #----------------------------------------------------------

    # Create the unified results (means of the diferents nodes)
    #----------------------------------------------------------
    results = {}

    for category_name in data[0]:

        category = {}

        for subcategory_name in data[0][category_name]:
            
            subcategory = []

            for i in range(len(data[0][category_name][subcategory_name])):
                step = []
                for d in data:
                    step.append(d[category_name][subcategory_name][i])
                subcategory.append(np.mean(step))
            
            category.update( {subcategory_name: subcategory} )

        results.update( {category_name: category} )
    #print(results)

    with open(f'{path_processed}/results.json', 'w') as file:
        json.dump(results, file, indent=4)
    #----------------------------------------------------------

