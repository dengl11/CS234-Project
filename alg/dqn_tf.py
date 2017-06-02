from config import *
import tensorflow as tf
from util import *
import os



class DQNTF(object):
    """
    Abstract Class for implementing a Q Network
    """


    def __init__(self, input_size, hidden_size, output_size, params, logger=None):
        """
        Initialize Q Network

        Args:
            logger: logger instance from logging module
        """
        config = DQNTfConfig
        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)
            
        # store hyper params
        self.logger = logger if logger else get_logger(config.log_path)

        self.gamma = params.get('gamma', 0.9)
        self.learning_rate = params.get('learning_rate', 1e-3)
        self.beta = params.get('beta', 0.9)
        self.grad_clip = params.get('grad_clip', -1e-3)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.build_model()
        

    def build_model(self):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.states      = tf.placeholder(tf.float32, shape=(None, self.input_size))
        self.next_states = tf.placeholder(tf.float32, shape=(None, self.input_size))
        self.done_mask   = tf.placeholder(tf.bool, shape=(None))
        self.rewards     = tf.placeholder(tf.float32, shape=(None))
        self.actions     = tf.placeholder(tf.int32, shape=(None))
        self.q           = self.get_q_values(self.states, "q")
        self.target_q    = self.get_q_values(self.states, "target_q")
        self.loss        = self.get_loss(self.q, self.target_q)
        optimizer        = get_solver_adam(self.learning_rate, self.beta)
        self.train_step  = optimizer.minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())


    def get_loss(self, q, target_q):
        """
        Input:
            q:          (tf tensor) shape = (batch_size, output_size)
            target_q:  (tf tensor) shape = (batch_size, output_size)
        Output:
            scaler loss
        """
        not_done = 1 - tf.cast(self.done_mask, tf.float32) # [batch_size]
        max_q = tf.reduce_max(target_q, axis = 1) # [batch_size]
        gamma = self.gamma
        Q_samp = self.rewards + gamma * tf.multiply(max_q, not_done) # [batch_size]
        q_extracted = tf.reduce_sum(tf.multiply(tf.one_hot(indices=self.actions, depth=self.output_size), q), axis=1)
        with tf.variable_scope("loss"):
            return tf.reduce_mean(tf.square(q_extracted - Q_samp)) # scalar
        



    def get_q_values(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, 252)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        hidden_size, output_size = self.hidden_size, self.output_size
        with tf.variable_scope(scope):
            x = state
            x = tf.layers.dense(x, hidden_size, activation=tf.nn.relu)
            x = tf.layers.dense(x, output_size, activation=tf.nn.relu)
            return x


    def train_batch(self, batch_experience):
        """
        Input:
            batch_experience: list of batch_size [(state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over)]
        Output:
            scaler loss
        """
        curr_states = np.concatenate([x[0] for x in batch_experience], axis = 0) # [batch_size, 252]
        next_states = np.concatenate([x[3] for x in batch_experience], axis = 0) # [batch_size, 252]
        done        = np.array([x[4] for x in batch_experience]) # [batch_size, 252]
        rewards     = np.array([x[2] for x in batch_experience]) # [batch_size, 252]
        actions     = np.array([x[1] for x in batch_experience]) # [batch_size, 252]
        dic = {
            self.states:        curr_states,
            self.next_states:   next_states,
            self.done_mask:     done,
            self.rewards:       rewards,
            self.actions:       actions
        }
        self.sess.run(self.train_step, dic)
        loss = self.sess.run(self.loss, dic)
        return loss


    def predict_action():
        """
        """
        pass