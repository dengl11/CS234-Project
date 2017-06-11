class AgentConfig:
	sys_request_slots = ['destination1', 'flightDate2', 'flightDate1', 'origin1', 'travelers']
	sys_inform_slots = ['destination1', 'flightDate2', 'flightDate1', 'origin1', 'travelers', 'taskcomplete', 'ticket']
	start_dia_acts = {
    'request':['destination1', 'flightDate2', 'flightDate1', 'origin1', 'travelers', 'ticket']
	}
	################################################################################
	#   A Basic Set of Feasible actions to be Consdered By an RL agent
	################################################################################
	feasible_actions = [
	    ############################################################################
	    #   greeting actions
	    ############################################################################
	    #{'diaact':"greeting", 'inform_slots':{}, 'request_slots':{}},
	    ############################################################################
	    #   confirm_question actions
	    ############################################################################
	    {'diaact':"confirm_question", 'inform_slots':{}, 'request_slots':{}},
	    ############################################################################
	    #   confirm_answer actions
	    ############################################################################
	    {'diaact':"confirm_answer", 'inform_slots':{}, 'request_slots':{}},
	    ############################################################################
	    #   thanks actions
	    ############################################################################
	    {'diaact':"thanks", 'inform_slots':{}, 'request_slots':{}},
	    ############################################################################
	    #   deny actions
	    ############################################################################
	    {'diaact':"deny", 'inform_slots':{}, 'request_slots':{}},
	]

	############################################################################
	#   Adding the inform actions
	############################################################################
	for slot in sys_inform_slots:
	    feasible_actions.append({'diaact':'inform', 'inform_slots':{slot:"PLACEHOLDER"}, 'request_slots':{}})

	############################################################################
	#   Adding the request actions
	############################################################################
	for slot in sys_request_slots:
	    feasible_actions.append({'diaact':'request', 'inform_slots':{}, 'request_slots': {slot: "UNK"}})
