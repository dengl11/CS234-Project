import numpy as np
from .kb_helper import *
from config import *
import random, copy

class StateTracker:
    """ 
    The state tracker maintains a record of which request slots are filled and which inform slots are filled 
    """
    def __init__(self, act_set, slot_set, movie_dictionary):
        self.act_set = act_set
        self.movie_dictionary = movie_dictionary
        self.slot_set = slot_set
        self.slot_set = slot_set
        self.hist_dic = None
        self.curr_slots = None
        self.init_episode()
        self.act_dim = 10
        self.kb_result_dim = 10
        self.turn_count = 0
        self.kb_helper = KBHelper(movie_dictionary)
    
    def init_episode(self):
        """ 
        Initialize a new episode (dialog), flush the current state and tracked slots 
        """
        self.act_dim = 10
        self.hist_vectors = np.zeros(self.act_dim)
        self.hist_dic = []
        self.turn_count = 0
        self.curr_slots = {}
        self.curr_slots['inform_slots'] = {}
        self.curr_slots['request_slots'] = {}
        self.curr_slots['proposed_slots'] = {}
        self.curr_slots['agt_request_slots'] = {}
    
    def get_state_for_agent(self):
        """ 
        Get the state representatons to send to agent 
        state = 
            {'user_action': self.hist_dic[-1], 
             'curr_slots': self.curr_slots, 
             'kb_results': self.kb_results_for_state()}
        """
        state = {
            "user_action": self.hist_dic[-1],
            "curr_slots":  self.curr_slots,
            "kb_results":  self.kb_helper.database_results_for_agent(self.curr_slots),
            'turn': self.turn_count, 
            'history': self.hist_dic, 
            'agent_action': self.hist_dic[-2] if len(self.hist_dic) > 1 else None
        }
        return copy.deepcopy(state)
    
    def dialog_history_vectors(self):
        """ Return the dialog history (both user and agent actions) in vector representation """
        return self.hist_vec

    def dialog_history_dictionaries(self):
        """  Return the dictionary representation of the dialog history (includes values) """
        return self.hist_dic
    
    
    def update(self, agent_action=None, user_action=None):
        """ 
        Update the state based on the latest action 
        """

        ########################################################################
        #  Make sure that the function was called properly
        ########################################################################
        assert(not (user_action and agent_action))
        assert(user_action or agent_action)

        ########################################################################
        #   Update state to reflect a new action by the agent.
        ########################################################################
        if agent_action:
            
            ####################################################################
            #   Handles the act_slot response (with values needing to be filled)
            ####################################################################
            if agent_action['act_slot_response']:
                response = copy.deepcopy(agent_action['act_slot_response'])
                
                inform_slots = self.kb_helper.fill_inform_slots(response['inform_slots'], self.curr_slots) # TODO this doesn't actually work yet, remove this warning when kb_helper is functional
                agent_action_values = {'turn': self.turn_count, 'speaker': "agent", 'diaact': response['diaact'], 'inform_slots': inform_slots, 'request_slots':response['request_slots']}
                
                agent_action['act_slot_response'].update({'diaact': response['diaact'], 'inform_slots': inform_slots, 'request_slots':response['request_slots'], 'turn':self.turn_count})
                
            elif agent_action['act_slot_value_response']:
                agent_action_values = copy.deepcopy(agent_action['act_slot_value_response'])
                # print("Updating state based on act_slot_value action from agent")
                agent_action_values['turn'] = self.turn_count
                agent_action_values['speaker'] = "agent"
                
            ####################################################################
            #   This code should execute regardless of which kind of agent produced action
            ####################################################################
            for slot in agent_action_values['inform_slots'].keys():
                self.curr_slots['proposed_slots'][slot] = agent_action_values['inform_slots'][slot]
                self.curr_slots['inform_slots'][slot] = agent_action_values['inform_slots'][slot] # add into inform_slots
                if slot in self.curr_slots['request_slots'].keys():
                    del self.curr_slots['request_slots'][slot]

            for slot in agent_action_values['request_slots'].keys():
                if slot not in self.curr_slots['agt_request_slots']:
                    self.curr_slots['agt_request_slots'][slot] = "UNK"

            self.hist_dic.append(agent_action_values)
            current_agent_vector = np.ones((1, self.act_dim))
            self.hist_vectors = np.vstack([self.hist_vectors, current_agent_vector])
                            
        ########################################################################
        #   Update the state to reflect a new action by the user
        ########################################################################
        elif user_action:
            
            ####################################################################
            #   Update the current slots
            ####################################################################
            for slot in user_action['inform_slots'].keys():
                self.curr_slots['inform_slots'][slot] = user_action['inform_slots'][slot]
                if slot in self.curr_slots['request_slots'].keys():
                    del self.curr_slots['request_slots'][slot]

            for slot in user_action['request_slots'].keys():
                if slot not in self.curr_slots['request_slots']:
                    self.curr_slots['request_slots'][slot] = "UNK"
            
            self.hist_vectors = np.vstack([self.hist_vectors, np.zeros((1,self.act_dim))])
            new_move = {'turn': self.turn_count, 'speaker': "user", 'request_slots': user_action['request_slots'], 'inform_slots': user_action['inform_slots'], 'diaact': user_action['diaact']}
            self.hist_dic.append(copy.deepcopy(new_move))

        ########################################################################
        #   This should never happen if the asserts passed
        ########################################################################
        else:
            pass

        ########################################################################
        #   This code should execute after update code regardless of what kind of action (agent/user)
        ########################################################################
        self.turn_count += 1
