import cPickle as pickle
import json
import sys

goal_file_path = sys.argv[1]

with open(goal_file_path, 'rb') as f:
    thawed = pickle.load(f)

with open(goal_file_path+'.json', 'w') as outfile:
    json.dump(thawed, outfile)

