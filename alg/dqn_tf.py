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
        self.config = DQNTfConfig
        self.batch_size = params['batch_size'] 

        # directory for training outputs
        if not os.path.exists(self.config.output_path):
            os.makedirs(self.config.output_path)
            
        # store hyper params
        # self.logger = logger if logger else get_logger(self.config.log_path)

        self.gamma = params.get('gamma', 0.9)
        self.learning_rate = self.config.lr
        self.beta = params.get('beta', 0.999)

        self.input_size = input_size # 272
        self.hidden_size = hidden_size # 60
        self.output_size = output_size # 43

        self.build_model()
        # self.saver = tf.train.Saver()


              
    def init_weight(self, fan_in, fan_out, name, scale = 1):
        scaler = np.sqrt(6./(fan_in + fan_out))
        init = scaler * (tf.random_uniform([fan_in, fan_out])*2 - 1)
        init *= scale
        return tf.Variable(init, name)

    def print_weight(self):
        w1_val, w2_val = self.sess.run([tf.norm(self.W1), tf.norm(self.W2)])
        # print("w1_val={} | w2_val={}".format(w1_val, w2_val))


    def build_model(self):
        tf.reset_default_graph()
        self.sess        = tf.Session()
        self.states      = tf.placeholder(tf.float32, shape=(None, self.input_size))
        self.next_states = tf.placeholder(tf.float32, shape=(None, self.input_size))
        self.done_mask   = tf.placeholder(tf.bool, shape=(None))
        self.rewards     = tf.placeholder(tf.float32, shape=(None))
        self.actions     = tf.placeholder(tf.int32, shape=(None))
        self.q, self.W1, self.W2    = self.get_q_values(self.states, "q")
        self.target_q, _, _    = self.get_q_values(self.next_states, "target_q")

        self.loss, self.reg_loss        = self.get_loss(self.q, self.target_q)
        
        self.set_train_step("q")
        self.set_update_step("q", "target_q")
        self.sess.run(tf.global_variables_initializer())
        

    


    def set_train_step(self, scope):
        trainable_var_key = tf.GraphKeys.TRAINABLE_VARIABLES
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, epsilon = 1e-8, momentum = 0.1)
        xs = tf.get_collection(key=trainable_var_key, scope=scope)
        with tf.variable_scope("loss"):
            grad_var_list = optimizer.compute_gradients(self.loss, xs)
            gradients = [x[0] for x in grad_var_list]
            if self.config.grad_clip:
                gradients = [tf.clip_by_norm(g, self.config.clip_val) for g in gradients]
            self.train_step = optimizer.apply_gradients([(gradients[i], xs[i]) for i in range(len(xs))])


    def set_update_step(self, q_scope, target_q_scope):
        trainable_var_key = tf.GraphKeys.TRAINABLE_VARIABLES
        q = tf.get_collection(key=trainable_var_key, scope=q_scope)
        q_target = tf.get_collection(key=trainable_var_key, scope=target_q_scope)
        all_assignments = [tf.assign(q_target[i], q[i]) for i in range(len(q))]
        self.update_step = tf.group(*all_assignments)



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
        W1, W2 = self.W1, self.W2
        reg = self.config.reg
        print(self.batch_size)
        with tf.variable_scope("loss") as scope:
            loss = tf.reduce_mean(tf.square(q_extracted - Q_samp)) # scalar
            # print("reg:", reg)
            reg_loss = 0.5 * reg * tf.square(tf.norm(W1))/self.batch_size
            reg_loss += 0.5 * reg * tf.square(tf.norm(W2))/self.batch_size
            # print(self.batch_size)
            return loss + reg_loss, reg_loss
        
    def get_q_values(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, 252)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns: gre
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        input_size, hidden_size, output_size = self.input_size, self.hidden_size, self.output_size
        with tf.variable_scope(scope):
            x = state
            w1 = self.init_weight(input_size, hidden_size, "W1")
            b1 = tf.Variable(tf.zeros(hidden_size), "b1")
            x = tf.add(tf.matmul(x, w1), b1)
            x = tf.nn.relu(x)

            w2 = self.init_weight(hidden_size, output_size, "W2", scale = 0.1)
            b2 = tf.Variable(tf.zeros(output_size), "b2")
            x = tf.add(tf.matmul(x, w2), b2)
            return x, w1, w2


    def train_batch(self, batch_experience):
        """
        Input:
            batch_experience: list of batch_size [(state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over)]
        Output:
            scaler loss
        """
        self.batch_size = len(batch_experience)
        curr_states = np.concatenate([x[0] for x in batch_experience], axis = 0) # [batch_size, 252]
        next_states = np.concatenate([x[3] for x in batch_experience], axis = 0) # [batch_size, 252]
        done        = np.array([x[4] for x in batch_experience]) # [batch_size, 252]
        rewards     = np.array([x[2] for x in batch_experience]) # [batch_size, 252]
        actions     = np.array([x[1] for x in batch_experience]) # [batch_size, 252]
        # print("rewards:", rewards)
        dic = {
            self.states:        curr_states,
            self.next_states:   next_states,
            self.done_mask:     done,
            self.rewards:       rewards,
            self.actions:       actions
        }
        self.sess.run([self.train_step], dic)
        loss, reg_loss = self.sess.run([self.loss, self.reg_loss], dic)
        # print("loss = {} | reg_loss = {}".format(loss, reg_loss))
        return loss



    def update_target_params(self):
        print("=============== update_target_params ===============")
        self.sess.run(self.update_step)
        
        
    def predict_action(self, states):
        """
        Input:
            states: [batch_size, ?]
        Output:
            actions: [batch_size]
        """
        action_values = self.sess.run(self.q, feed_dict={self.states: states})[0]
        # print("action_values=", action_values)
        actions = np.argmax(action_values)
        # print("actions: ", actions)
        return actions

    def save(self):
        """
        Saves session
        """
        if not os.path.exists(self.config.model_output):
            os.makedirs(self.config.model_output)

        self.saver.save(self.sess, self.config.model_output)

