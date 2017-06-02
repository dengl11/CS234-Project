import argparse, json
import matplotlib.pyplot as plt


def read_performance_records(path):
    """ load the performance score (.json) file """
    
    data = json.load(open(path, 'rb'))
    for key in data['success_rate'].keys():
        if int(key) > -1:
            print("%s\t%s\t%s\t%s" % (key, data['success_rate'][key], data['ave_turns'][key], data['ave_reward'][key]))
            

def load_performance_file(path):
    """ load the performance score (.json) file """
    
    data = json.load(open(path, 'rb'))
    numbers = {'x': [], 'success_rate':[], 'ave_turns':[], 'ave_rewards':[]}
    keylist = [int(key) for key in data['success_rate'].keys()]
    keylist.sort()

    for key in keylist:
        if int(key) > -1:
            numbers['x'].append(int(key))
            numbers['success_rate'].append(data['success_rate'][str(key)])
            numbers['ave_turns'].append(data['ave_turns'][str(key)])
            numbers['ave_rewards'].append(data['ave_reward'][str(key)])
    return numbers

def draw_learning_curve(numbers, size = (24, 8)):
    """ draw the learning curve """
    plt.figure(figsize=size)
    plt.xlabel('Simulation Epoch', fontSize=18)
    plt.ylabel('Success Rate', fontSize=18)
    plt.title('Learning Curve', fontSize=20)
    plt.grid(True)

    plt.plot(numbers['x'], numbers['success_rate'], 'r', lw=1)
    plt.show()