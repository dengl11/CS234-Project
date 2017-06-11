from alg import *
from agent import *

class DQNAgentTF(AgentDQN):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None, transfer=False, path=None):
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())
        
        self.feasible_actions = AgentConfig.feasible_actions
        self.num_actions = len(self.feasible_actions)
        
        self.epsilon = params['epsilon']
        self.agent_run_mode = params['agent_run_mode']

        self.agent_act_level = params['agent_act_level']
        self.clear_exp_pool()
        
        self.experience_replay_pool_size = params.get('experience_replay_pool_size', 1000)
        self.hidden_size = params.get('dqn_hidden_size', 60)
        self.gamma = params.get('gamma', 0.9)
        self.predict_mode = params.get('predict_mode', False)
        self.warm_start = params.get('warm_start', 0)
        
        self.max_turn = params['max_turn'] + 4
        self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn
        
        self.transfer = transfer
        if self.transfer:
            self.model = DQNTF_all(self.state_dimension, self.hidden_size, self.num_actions, params, path=path)
        else :
            self.model = DQNTF(self.state_dimension, self.hidden_size, self.num_actions, params)

        # self.dqn = DQN(self.state_dimension, self.hidden_size, self.num_actions)
        # self.clone_dqn = copy.deepcopy(self.dqn)
        self.cur_bellman_err = 0
                
        # Prediction Mode: load trained DQN model
        if params['trained_model_path'] != None:
            self.dqn.model = copy.deepcopy(self.load_trained_DQN(params['trained_model_path']))
            self.clone_dqn = copy.deepcopy(self.dqn)
            self.predict_mode = True
            self.warm_start = 2


    def run_policy(self, representation):
        """ epsilon-greedy policy """
        
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            if self.warm_start == 1:
                if len(self.experience_replay_pool) > self.experience_replay_pool_size:
                    self.warm_start = 2
                return self.rule_policy()
            else:
                # return self.dqn.predict(representation, {}, predict_model=True)
                return self.model.predict_action(representation)
    
    
    def train(self, batch_size=1, num_batches=100, show_every=1):
        """ Train DQN with experience replay """
        assert len(self.experience_replay_pool)>0, "No Experience Replay!"
        print("Train on : {}".format(len(self.experience_replay_pool)))
        for iter_batch in range(1, num_batches+1):
            self.cur_bellman_err = 0
            for iter in range(len(self.experience_replay_pool)//(batch_size)):
                batch = [random.choice(self.experience_replay_pool) for i in range(batch_size)]
                loss = self.model.train_batch(batch)
                # batch_struct = self.dqn.singleBatch(batch, {'gamma': self.gamma}, self.clone_dqn)
                self.cur_bellman_err += loss
                # self.cur_bellman_err += batch_struct['cost']['total_cost']
            # after every batch update target q
            # self.model.update_target_params()
            if iter_batch%show_every==0: print("- cur bellman err %.4f, experience replay pool %s" % (float(self.cur_bellman_err)/len(self.experience_replay_pool), len(self.experience_replay_pool)))
        # print(batch_struct['cost'])
        return self.cur_bellman_err
            # if iter_batch%DQNTfConfig.save_every==0: 
            #     self.model.save()
            #     print("Model saved!")
            
            
    