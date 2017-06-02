class UserConfig:
	################################################################################
	#  Constraint Check
	################################################################################
	CONSTRAINT_CHECK_FAILURE = 0
	CONSTRAINT_CHECK_SUCCESS = 1

	start_dia_acts = {
    	#'greeting':[],
    	'request':['moviename', 'starttime', 'theater', 'city', 'state', 'date', 'genre', 'ticket', 'numberofpeople']
	}