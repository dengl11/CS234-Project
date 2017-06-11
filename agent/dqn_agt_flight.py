from .agent import *
from config import *
from alg import *
import copy, argparse, json, random
import numpy as np
from six.moves import cPickle as pickle


class AgentDQN(Agent):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())
        
        self.feasible_actions = AgentConfig.feasible_actions
        self.num_actions = len(self.feasible_actions)
        
        self.epsilon = params['epsilon']
        self.agent_run_mode = params['agent_run_mode']
        print("agent_run_mode:{}".format(self.agent_run_mode))
        self.agent_act_level = params['agent_act_level']
        self.clear_exp_pool()
        
        self.experience_replay_pool_size = params.get('experience_replay_pool_size', 1000)
        self.hidden_size = params.get('dqn_hidden_size', 60)
        self.gamma = params.get('gamma', 0.9)
        self.predict_mode = params.get('predict_mode', False)
        self.warm_start = params.get('warm_start', 0)
        
        self.max_turn = params['max_turn'] + 4
        self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn
        self.state_dimension = 300
        
        self.dqn = DQN(self.state_dimension, self.hidden_size, self.num_actions)
        self.clone_dqn = copy.deepcopy(self.dqn)
        
        self.cur_bellman_err = 0
                
        # Prediction Mode: load trained DQN model
        if params['trained_model_path'] != None:
            self.dqn.model = copy.deepcopy(self.load_trained_DQN(params['trained_model_path']))
            self.clone_dqn = copy.deepcopy(self.dqn)
            self.predict_mode = True
            self.warm_start = 2
            
    def clear_exp_pool(self): 
        self.experience_replay_pool = [] #experience replay pool <s_t, a_t, r_t, s_t+1>
        
    def init_episode(self):
        """ Initialize a new episode. This function is called every time a new episode is run. """
        
        self.current_slot_id = 0
        self.phase = 0
        self.request_set = ['destination1', 'flightDate2', 'flightDate1', 'origin1', 'travelers']
        # self.request_set = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']
    
    
    def state_to_action(self, state):
        """ DQN: Input state, output action """
        
        self.representation = self.prepare_state_representation(state)
        self.action = self.run_policy(self.representation)
        act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}
        
    
    def prepare_state_representation(self, state):
        """ Create the representation for each state """
        user_action = state['user_action']
        current_slots = state['curr_slots']
        kb_results_dict = state['kb_results']
        agent_last = state['agent_action']
        
        ########################################################################
        #   Create one-hot of acts to represent the current user action
        ########################################################################
        user_act_rep =  np.zeros((1, self.act_cardinality))
        user_act_rep[0,self.act_set[user_action['diaact']]] = 1.0

        ########################################################################
        #     Create bag of inform slots representation to represent the current user action
        ########################################################################
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['inform_slots'].keys():
            user_inform_slots_rep[0,self.slot_set[slot]] = 1.0

        ########################################################################
        #   Create bag of request slots representation to represent the current user action
        ########################################################################
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['request_slots'].keys():
            user_request_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Creat bag of filled_in slots based on the current_slots
        ########################################################################
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in current_slots['inform_slots']:
            current_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent act
        ########################################################################
        agent_act_rep = np.zeros((1,self.act_cardinality))
        if agent_last:
            agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0

        ########################################################################
        #   Encode last agent inform slots
        ########################################################################
        agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['inform_slots'].keys():
                agent_inform_slots_rep[0,self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent request slots
        ########################################################################
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['request_slots'].keys():
                agent_request_slots_rep[0,self.slot_set[slot]] = 1.0
        
        turn_rep = np.zeros((1,1)) + state['turn'] / 10.

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        ########################################################################
        #   Representation of KB results (scaled counts)
        ########################################################################
        kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + kb_results_dict['matching_all_constraints'] / 100.
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.

        ########################################################################
        #   Representation of KB results (binary)
        ########################################################################
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum( kb_results_dict['matching_all_constraints'] > 0.)
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_binary_rep[0, self.slot_set[slot]] = np.sum( kb_results_dict[slot] > 0.)

        self.final_representation = np.hstack([user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep, agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep])
        state = np.zeros((1, 300))
        state[0, 0:self.final_representation.shape[1]] = self.final_representation
        self.final_representation = state
        return self.final_representation
      
    def run_policy(self, representation):
        """ 
        epsilon-greedy policy 
        Input: 
            representation: (1, 252)
        """


        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            if self.warm_start == 1:
                if len(self.experience_replay_pool) > self.experience_replay_pool_size:
                    self.warm_start = 2
                return self.rule_policy()
            else:
                return self.dqn.predict(representation, {}, predict_model=True)
    
    def rule_policy(self):
        """ Rule Policy """
        
        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"}, 'request_slots': {} }
            self.phase += 1
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {} }
                
        return self.action_index(act_slot_response)
    
    def action_index(self, act_slot_response):
        """ Return the index of action """
        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        raise Exception("action index not found")
        return None
    
    
    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over):
        """ Register feedback from the environment, to be stored as future training data """
        
        state_t_rep = self.prepare_state_representation(s_t)
        action_t = self.action
        reward_t = reward
        state_tplus1_rep = self.prepare_state_representation(s_tplus1)
        training_example = (state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over)
        
        if self.predict_mode == False: # Training Mode
            if self.warm_start == 1:
                self.experience_replay_pool.append(training_example)
        else: # Prediction Mode
            self.experience_replay_pool.append(training_example)
    
    def train(self, batch_size=1, num_batches=100, show_every=1):
        """ Train DQN with experience replay """
        assert len(self.experience_replay_pool)>0, "No Experience Replay!"
        print("Train on : {}".format(len(self.experience_replay_pool)))
        for iter_batch in range(1, num_batches+1):
            self.cur_bellman_err = 0
            for iter in range(len(self.experience_replay_pool)//(batch_size)):
                batch = [random.choice(self.experience_replay_pool) for i in range(batch_size)]
                batch_struct = self.dqn.singleBatch(batch, {'gamma': self.gamma}, self.clone_dqn)
                self.cur_bellman_err += batch_struct['cost']['total_cost']
            
            if iter_batch%show_every==0: print("cur bellman err [%.4f], experience replay pool %s" % (float(self.cur_bellman_err)/len(self.experience_replay_pool), len(self.experience_replay_pool)))
        
        return self.cur_bellman_err
            
            
    ################################################################################
    #    Debug Functions
    ################################################################################
    def save_experience_replay_to_file(self, path):
        """ Save the experience replay pool to a file """
        try:
            pickle.dump(self.experience_replay_pool, open(path, "wb"))
            print('saved model in %s' % (path, ))
        except(Exception, e):
            print('Error: Writing model fails: %s' % (path, ))
            print(e         )
    
    def load_experience_replay_from_file(self, path):
        """ Load the experience replay pool from a file"""
        self.experience_replay_pool = pickle.load(open(path, 'rb'))
    
             
    def load_trained_DQN(self, path):
        """ Load the trained DQN from a file """
        
        trained_file = pickle.load(open(path, 'rb'), encoding="latin1")
        model = trained_file['model']
        
        print("trained DQN Parameters:", json.dumps(trained_file['params'], indent=2))
        return(model)
