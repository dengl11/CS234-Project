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

def draw_learning_curve(numbers, size = (16, 8)):
    """ draw the learning curve """
    hfont = {'fontsize':24}
    plt.figure(figsize=size)
    plt.xlabel('Simulation Epoch', fontSize=18, labelpad=20)
    plt.ylabel('Success Rate', fontSize=18, labelpad=20)
    
    plt.title('Learning Curve',**hfont, y=1.1)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.grid(False)

    ax = plt.plot(numbers, 'r', lw=4)
    # plt.plot(numbers['x'], numbers['success_rate'], 'r', lw=1)
    plt.show()


def draw_loss_curve(numbers, size = (16, 8)):
    plt.figure(figsize=size)
    hfont = {'fontsize':24}
    plt.xlabel('Simulation Epoch', fontSize=18, labelpad=20)
    plt.ylabel('Loss', fontSize=18, labelpad=20)
    plt.title('Loss Curve', **hfont, y=1.1)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.grid(False)

    plt.plot(numbers, 'b', lw=4)
    # plt.plot(numbers['x'], numbers['success_rate'], 'r', lw=1)
    plt.show()