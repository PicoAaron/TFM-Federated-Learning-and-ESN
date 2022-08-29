import json
import os
import numpy as np
import matplotlib.pyplot as plt

def process(num_experiment):
    path = f'./results/experiment_results/experiment_{num_experiment}'
    path_processed = f'./results/processed_results/'
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

    with open(f'{path_processed}/results_{num_experiment}.json', 'w') as file:
        json.dump(results, file, indent=4)

    with open(f'{path}/results_{num_experiment}.json', 'w') as file:
        json.dump(results, file, indent=4)
    #----------------------------------------------------------

    # Image
    #----------------------------------------------------------

    plt.plot(results['Federated']['val_loss'])
    plt.title('Federated Learning: Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'results/processed_results/results_federated.png')

    #----------------------------------------------------------

    #plt.plot(results['Federated']['val_loss'])
    plt.plot(results['No Federated (Local)']['val_loss'])
    plt.title('Model loss: FL vs ML')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Federated Learning', 'Local Machine Learning'], loc='upper left')
    plt.savefig(f'results/processed_results/results_FL_ML.png')

if __name__ == "__main__":

    process()
    

