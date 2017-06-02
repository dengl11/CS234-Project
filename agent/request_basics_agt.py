from .agent import *
from config import *

class RequestBasicsAgent(Agent):
    """ 
    A simple agent to test the system. 
    This agent should simply request all the basic slots and then issue: thanks(). 
    """
    def init_episode(self):
        self.state = {}
        self.state['diaact'] = 'UNK'
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['turn'] = -1
        self.current_slot_id = 0
        self.request_set = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']
        self.phase = 0

    def state_to_action(self, state):
        """ Run current policy on state and produce an action """
        self.state['turn'] += 2
        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}
            act_slot_response['turn'] = self.state['turn']
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"}, 'request_slots': {}, 'turn':self.state['turn']}
            self.phase += 1
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}, 'turn': self.state['turn']}
        else:
            raise Exception("THIS SHOULD NOT BE POSSIBLE (AGENT CALLED IN UNANTICIPATED WAY)")
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}