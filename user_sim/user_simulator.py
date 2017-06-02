class UserSimulator:
    def set_nlg_model(self, nlg_model):
        self.nlg_model = nlg_model  
    
    def set_nlu_model(self, nlu_model):
        self.nlu_model = nlu_model
    
    def add_nl_to_action(self, user_action):
        """ 
        Add NL to User Dia_Act 
        """
        
        user_nlg_sentence = self.nlg_model.convert_diaact_to_nl(user_action, 'usr')
        user_action['nl'] = user_nlg_sentence
        
        if self.act_level == 1:
            user_nlu_res = self.nlu_model.generate_dia_act(user_action['nl']) # NLU
            if user_nlu_res != None:
                #user_nlu_res['diaact'] = user_action['diaact'] # or not?
                user_action.update(user_nlu_res)
    