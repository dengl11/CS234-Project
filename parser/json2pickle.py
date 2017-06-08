import cPickle as pickle
import jsonpickle
import sys

goal_file_path = sys.argv[1]

with open(goal_file_path, 'rb') as f:
    data = ' '.join(f.readlines())
    thawed = jsonpickle.decode(data)

print thawed

with open(goal_file_path + ".p", 'wb') as f:
    pickle.dump(thawed, f)

