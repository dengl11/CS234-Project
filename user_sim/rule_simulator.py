from .user_simulator import *
from config import *
import numpy as np
import random, copy

class RuleSimulator(UserSimulator):
    """ 
    A rule-based user simulator for testing dialog policy 
    """
    def __init__(self, movie_dict = None, act_set = None, slot_set = None, start_set = None, params = None):
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set
        self.max_turn = params['max_turn']
        self.slot_err_prob = params['slot_err_prob']
        self.slot_err_mode = params['slot_err_mode']
        self.intent_err_prob = params['intent_err_prob']
        self.run_mode = params['run_mode']
        self.act_level = params['act_level']
        self.learn_phase = params['learn_phase']
     
    
    def _sample_goal(self, goal_set):
        return np.random.choice(self.start_set[self.learn_phase])
        
    def init_episode(self):
        """ 
        Initialize a new episode (dialog) 
            state['hist_slots']: keeps all the informed_slots
            state['rest_slots']: keep all the slots (which is still in the stack yet)
        """
        self.state = {}
        self.state['hist_slots'] = {}
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['rest_slots'] = []
        self.state['turn'] = 0
        self.episode_done = False
        self.status = UserSimulatorConfig.NO_OUTCOME_YET
        
        self.goal = self._sample_goal(self.start_set)
        self.goal['request_slots']['ticket'] = 'UNK'
        self.constraint_check = DlgManagerConfig.CONSTRAINT_CHECK_FAILURE
        
        # sample first action
        user_action = self._sample_action()
        assert (self.episode_done != 1),' but we just started'
        return user_action

    def _sample_action(self):
        """ 
        randomly sample a start action based on user goal 
        """
        
        self.state['diaact'] = random.choice(list(UserConfig.start_dia_acts.keys()))
        
        # "sample" informed slots
        if len(self.goal['inform_slots']) > 0:
            known_slot = random.choice(list(self.goal['inform_slots'].keys()))
            self.state['inform_slots'][known_slot] = self.goal['inform_slots'][known_slot]

            if 'moviename' in self.goal['inform_slots'].keys(): # 'moviename' must appear in the first user turn
                self.state['inform_slots']['moviename'] = self.goal['inform_slots']['moviename']
                
            for slot in self.goal['inform_slots'].keys():
                if known_slot == slot or slot == 'moviename': continue
                self.state['rest_slots'].append(slot)
        
        self.state['rest_slots'].extend(self.goal['request_slots'].keys())
        
        # "sample" a requested slot
        request_slot_set = list(self.goal['request_slots'].keys())
        request_slot_set.remove('ticket')
        if len(request_slot_set) > 0:
            request_slot = random.choice(request_slot_set)
        else:
            request_slot = 'ticket'
        self.state['request_slots'][request_slot] = 'UNK'
        
        if len(self.state['request_slots']) == 0:
            self.state['diaact'] = 'inform'

        if (self.state['diaact'] in ['thanks','closing']): self.spisode_done = True
        else: self.spisode_done = False 

        sample_action = {}
        sample_action['diaact'] = self.state['diaact']
        sample_action['inform_slots'] = self.state['inform_slots']
        sample_action['request_slots'] = self.state['request_slots']
        sample_action['turn'] = self.state['turn']
        
        self.add_nl_to_action(sample_action)
        return sample_action
    
        
        
    def corrupt(self, user_action):
        """ Randomly corrupt an action with error probs (slot_err_probability and slot_err_mode) on Slot and Intent (intent_err_probability). """
        
        for slot in user_action['inform_slots'].keys():
            slot_err_prob_sample = random.random()
            if slot_err_prob_sample < self.slot_err_prob: # add noise for slot level
                if self.slot_err_mode == 0: # replace the slot_value only
                    if slot in self.movie_dict.keys(): user_action['inform_slots'][slot] = random.choice(self.movie_dict[slot])
                elif self.slot_err_mode == 1: # combined
                    slot_err_random = random.random()
                    if slot_err_random <= 0.33:
                        if slot in self.movie_dict.keys(): user_action['inform_slots'][slot] = random.choice(self.movie_dict[slot])
                    elif slot_err_random > 0.33 and slot_err_random <= 0.66:
                        del user_action['inform_slots'][slot]
                        random_slot = random.choice(self.movie_dict.keys())
                        user_action[random_slot] = random.choice(self.movie_dict[random_slot])
                    else:
                        del user_action['inform_slots'][slot]
                elif self.slot_err_mode == 2: #replace slot and its values
                    del user_action['inform_slots'][slot]
                    random_slot = random.choice(self.movie_dict.keys())
                    user_action[random_slot] = random.choice(self.movie_dict[random_slot])
                elif self.slot_err_mode == 3: # delete the slot
                    del user_action['inform_slots'][slot]   
        
    def next(self, system_action):
        """ Generate next User Action based on last System Action """
        
        self.state['turn'] += 2
        self.spisode_done = False
        self.dialog_status = UserSimulatorConfig.NO_OUTCOME_YET
        
        sys_act = system_action['diaact']
        
        if (self.max_turn > 0 and self.state['turn'] > self.max_turn):
            self.dialog_status = DlgManagerConfig.FAILED_DIALOG
            self.spisode_done = True
            self.state['diaact'] = "closing"
        else:
            self.state['hist_slots'].update(self.state['inform_slots'])
            self.state['inform_slots'].clear()

            if sys_act == "inform":
                self.response_inform(system_action)
            elif sys_act == "multiple_choice":
                self.response_multiple_choice(system_action)
            elif sys_act == "request":
                self.response_request(system_action) 
            elif sys_act == "thanks":
                self.response_thanks(system_action)
            elif sys_act == "confirm_answer":
                self.response_confirm_answer(system_action)
            elif sys_act == "closing":
                self.spisode_done = True
                self.state['diaact'] = "thanks"

        self.corrupt(self.state)
        
        response_action = {}
        response_action['diaact'] = self.state['diaact']
        response_action['inform_slots'] = self.state['inform_slots']
        response_action['request_slots'] = self.state['request_slots']
        response_action['turn'] = self.state['turn']
        response_action['nl'] = ""
        
        # add NL to dia_act
        self.add_nl_to_action(response_action)                       
        return response_action, self.spisode_done, self.dialog_status
    
    def response_confirm_answer(self, system_action):
        """ Response for Confirm_Answer (System Action) """
    
        if len(self.state['rest_slots']) > 0:
            request_slot = random.choice(self.state['rest_slots'])

            if request_slot in self.goal['request_slots'].keys():
                self.state['diaact'] = "request"
                self.state['request_slots'][request_slot] = "UNK"
            elif request_slot in self.goal['inform_slots'].keys():
                self.state['diaact'] = "inform"
                self.state['inform_slots'][request_slot] = self.goal['inform_slots'][request_slot]
                if request_slot in self.state['rest_slots']:
                    self.state['rest_slots'].remove(request_slot)
        else:
            self.state['diaact'] = "thanks"
            
    def response_thanks(self, system_action):
        """ Response for Thanks (System Action) """
        
        self.spisode_done = True
        self.dialog_status = DlgManagerConfig.SUCCESS_DIALOG

        request_slot_set = copy.deepcopy(list(self.state['request_slots'].keys()))
        if 'ticket' in request_slot_set:
            request_slot_set.remove('ticket')
        rest_slot_set = copy.deepcopy(self.state['rest_slots'])
        if 'ticket' in rest_slot_set:
            rest_slot_set.remove('ticket')

        if len(request_slot_set) > 0 or len(rest_slot_set) > 0:
            self.dialog_status = DlgManagerConfig.FAILED_DIALOG

        for info_slot in self.state['hist_slots'].keys():
            if self.state['hist_slots'][info_slot] == NlgConfig.NO_VALUE_MATCH:
                self.dialog_status = DlgManagerConfig.FAILED_DIALOG
            if info_slot in self.goal['inform_slots'].keys():
                if self.state['hist_slots'][info_slot] != self.goal['inform_slots'][info_slot]:
                    self.dialog_status = DlgManagerConfig.FAILED_DIALOG

        if 'ticket' in system_action['inform_slots'].keys():
            if system_action['inform_slots']['ticket'] == DlgManagerConfig.NO_VALUE_MATCH:
                self.dialog_status = DlgManagerConfig.FAILED_DIALOG
                
        if self.constraint_check == DlgManagerConfig.CONSTRAINT_CHECK_FAILURE:
            self.dialog_status = DlgManagerConfig.FAILED_DIALOG
    
    def response_request(self, system_action):
        """ Response for Request (System Action) """
        
        if len(system_action['request_slots'].keys()) > 0:
            slot = list(system_action['request_slots'].keys())[0] # only one slot
            if slot in self.goal['inform_slots'].keys(): # request slot in user's constraints  #and slot not in self.state['request_slots'].keys():
                self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
                self.state['diaact'] = "inform"
                if slot in self.state['rest_slots']: self.state['rest_slots'].remove(slot)
                if slot in self.state['request_slots'].keys(): del self.state['request_slots'][slot]
                self.state['request_slots'].clear()
            elif slot in self.goal['request_slots'].keys() and slot not in self.state['rest_slots'] and slot in self.state['hist_slots'].keys(): # the requested slot has been answered
                self.state['inform_slots'][slot] = self.state['hist_slots'][slot]
                self.state['request_slots'].clear()
                self.state['diaact'] = "inform"
            elif slot in self.goal['request_slots'].keys() and slot in self.state['rest_slots']: # request slot in user's goal's request slots, and not answered yet
                self.state['diaact'] = "request" # "confirm_question"
                self.state['request_slots'][slot] = "UNK"

                ########################################################################
                # Inform the rest of informable slots
                ########################################################################
                for info_slot in self.state['rest_slots']:
                    if info_slot in self.goal['inform_slots'].keys():
                        self.state['inform_slots'][info_slot] = self.goal['inform_slots'][info_slot]

                for info_slot in self.state['inform_slots'].keys():
                    if info_slot in self.state['rest_slots']:
                        self.state['rest_slots'].remove(info_slot)
            else:
                if len(self.state['request_slots']) == 0 and len(self.state['rest_slots']) == 0:
                    self.state['diaact'] = "thanks"
                else:
                    self.state['diaact'] = "inform"
                self.state['inform_slots'][slot] = NlgConfig.I_DO_NOT_CARE
        else: # this case should not appear
            if len(self.state['rest_slots']) > 0:
                random_slot = random.choice(self.state['rest_slots'])
                if random_slot in self.goal['inform_slots'].keys():
                    self.state['inform_slots'][random_slot] = self.goal['inform_slots'][random_slot]
                    self.state['rest_slots'].remove(random_slot)
                    self.state['diaact'] = "inform"
                elif random_slot in self.goal['request_slots'].keys():
                    self.state['request_slots'][random_slot] = self.goal['request_slots'][random_slot]
                    self.state['diaact'] = "request"

    def response_multiple_choice(self, system_action):
        """ Response for Multiple_Choice (System Action) """
        
        slot = system_action['inform_slots'].keys()[0]
        if slot in self.goal['inform_slots'].keys():
            self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
        elif slot in self.goal['request_slots'].keys():
            self.state['inform_slots'][slot] = random.choice(system_action['inform_slots'][slot])

        self.state['diaact'] = "inform"
        if slot in self.state['rest_slots']: self.state['rest_slots'].remove(slot)
        if slot in self.state['request_slots'].keys(): del self.state['request_slots'][slot]
        
    def response_inform(self, system_action):
        """ Response for Inform (System Action) """
        
        if 'taskcomplete' in system_action['inform_slots'].keys(): # check all the constraints from agents with user goal
            self.state['diaact'] = "thanks"
            #if 'ticket' in self.state['rest_slots']: self.state['request_slots']['ticket'] = 'UNK'
            self.constraint_check = DlgManagerConfig.CONSTRAINT_CHECK_SUCCESS
                    
            if system_action['inform_slots']['taskcomplete'] == NlgConfig.NO_VALUE_MATCH:
                self.state['hist_slots']['ticket'] = NlgConfig.NO_VALUE_MATCH
                if 'ticket' in self.state['rest_slots']: self.state['rest_slots'].remove('ticket')
                if 'ticket' in self.state['request_slots'].keys(): del self.state['request_slots']['ticket']
                    
            for slot in self.goal['inform_slots'].keys():
                #  Deny, if the answers from agent can not meet the constraints of user
                if slot not in system_action['inform_slots'].keys() or (self.goal['inform_slots'][slot].lower() != system_action['inform_slots'][slot].lower()):
                    self.state['diaact'] = "deny"
                    self.state['request_slots'].clear()
                    self.state['inform_slots'].clear()
                    self.constraint_check = DlgManagerConfig.CONSTRAINT_CHECK_FAILURE
                    break
        else:
            for slot in system_action['inform_slots'].keys():
                self.state['hist_slots'][slot] = system_action['inform_slots'][slot]
                        
                if slot in self.goal['inform_slots'].keys():
                    if system_action['inform_slots'][slot] == self.goal['inform_slots'][slot]:
                        if slot in self.state['rest_slots']: self.state['rest_slots'].remove(slot)
                                
                        if len(self.state['request_slots']) > 0:
                            self.state['diaact'] = "request"
                        elif len(self.state['rest_slots']) > 0:
                            rest_slot_set = copy.deepcopy(self.state['rest_slots'])
                            if 'ticket' in rest_slot_set:
                                rest_slot_set.remove('ticket')

                            if len(rest_slot_set) > 0:
                                inform_slot = random.choice(rest_slot_set) # self.state['rest_slots']
                                if inform_slot in self.goal['inform_slots'].keys():
                                    self.state['inform_slots'][inform_slot] = self.goal['inform_slots'][inform_slot]
                                    self.state['diaact'] = "inform"
                                    self.state['rest_slots'].remove(inform_slot)
                                elif inform_slot in self.goal['request_slots'].keys():
                                    self.state['request_slots'][inform_slot] = 'UNK'
                                    self.state['diaact'] = "request"
                            else:
                                self.state['request_slots']['ticket'] = 'UNK'
                                self.state['diaact'] = "request"
                        else: # how to reply here?
                            self.state['diaact'] = "thanks" # replies "closing"? or replies "confirm_answer"
                    else: # != value  Should we deny here or ?
                        ########################################################################
                        # TODO When agent informs(slot=value), where the value is different with the constraint in user goal, Should we deny or just inform the correct value?
                        ########################################################################
                        self.state['diaact'] = "inform"
                        self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
                        if slot in self.state['rest_slots']: self.state['rest_slots'].remove(slot)
                else:
                    if slot in self.state['rest_slots']:
                        self.state['rest_slots'].remove(slot)
                    if slot in self.state['request_slots'].keys():
                        del self.state['request_slots'][slot]

                    if len(self.state['request_slots']) > 0:
                        request_set = list(self.state['request_slots'].keys())
                        if 'ticket' in request_set:
                            request_set.remove('ticket')

                        if len(request_set) > 0:
                            request_slot = random.choice(request_set)
                        else:
                            request_slot = 'ticket'

                        self.state['request_slots'][request_slot] = "UNK"
                        self.state['diaact'] = "request"
                    elif len(self.state['rest_slots']) > 0:
                        rest_slot_set = copy.deepcopy(self.state['rest_slots'])
                        if 'ticket' in rest_slot_set:
                            rest_slot_set.remove('ticket')

                        if len(rest_slot_set) > 0:
                            inform_slot = random.choice(rest_slot_set) #self.state['rest_slots']
                            if inform_slot in self.goal['inform_slots'].keys():
                                self.state['inform_slots'][inform_slot] = self.goal['inform_slots'][inform_slot]
                                self.state['diaact'] = "inform"
                                self.state['rest_slots'].remove(inform_slot)
                                        
                                if 'ticket' in self.state['rest_slots']:
                                    self.state['request_slots']['ticket'] = 'UNK'
                                    self.state['diaact'] = "request"
                            elif inform_slot in self.goal['request_slots'].keys():
                                self.state['request_slots'][inform_slot] = self.goal['request_slots'][inform_slot]
                                self.state['diaact'] = "request"
                        else:
                            self.state['request_slots']['ticket'] = 'UNK'
                            self.state['diaact'] = "request"
                    else:
                        self.state['diaact'] = "thanks" # or replies "confirm_answer"