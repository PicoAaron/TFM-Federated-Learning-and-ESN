import json
import os
import numpy as np
import matplotlib.pyplot as plt


def join_results(data):
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
                    #print(i)
                    #print(d)
                    step.append(d[category_name][subcategory_name][i])
                subcategory.append(np.mean(step))
            
            category.update( {subcategory_name: subcategory} )

        results.update( {category_name: category} )
    
    return results


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
    
    results = join_results(data)

    with open(f'{path_processed}/results_{num_experiment}.json', 'w') as file:
        json.dump(results, file, indent=4)

    with open(f'{path}/results_{num_experiment}.json', 'w') as file:
        json.dump(results, file, indent=4)
    #----------------------------------------------------------

    # Image
    #----------------------------------------------------------
'''
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
'''

def average_results(path=f'./results/processed_results/', title_1='', title_2=''):
    #path = f'./results/processed_results/'

    files = os.listdir(path)

    # Read the results of the experiment
    #----------------------------------------------------------
    data = []

    for file_name in files:
        if file_name != 'results_average.json':
            try:
                with open(f'{path}/{file_name}') as file:
                    #data = json.load(file)
                    data.append(json.load(file))
            except:
                pass
    

    results = join_results(data)
    
    with open(f'{path}/results_average.json', 'w') as file:
        json.dump(results, file, indent=4)

    # Image
    #----------------------------------------------------------

    plt.plot(results['Federated']['val_loss'])
    plt.title(f'Federated Learning{title_1}: Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'{path}/results_federated.png')

    #----------------------------------------------------------
    
    plt.plot(results['Federated']['val_loss'])
    plt.plot(results['No Federated (Local)']['val_loss'])
    plt.title(f'Model loss: FL{title_1} vs ML{title_2}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Federated Learning', 'Local Machine Learning'], loc='upper left')
    plt.savefig(f'{path}/results_FL_ML.png')
    

if __name__ == "__main__":

    #process()

    title='ML_average'
    average_results(path=f'./results/FLvsML/{title}', title_1=f' ({title})', title_2=' (local)')
    

