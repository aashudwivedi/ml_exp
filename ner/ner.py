import sys
import numpy as np
import tensorflow as tf
import utils

sys.path.append("..")

from evaluation import precision_recall_f1

data = utils.Data()


class BiLSTMModel(object):

    def declare_placeholders(self):
        """Specifies placeholders for the model."""

        # Placeholders for input and ground truth output.
        self.input_batch = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_batch')
        self.ground_truth_tags = tf.placeholder(dtype=tf.int32, shape=[None, None], name='ground_truth_batch')

        # Placeholder for lengths of the sequences.
        self.lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='lengths')

        # Placeholder for a dropout keep probability. If we don't feed
        # a value for this placeholder, it will be equal to 1.0.
        self.dropout_ph = tf.placeholder_with_default(tf.cast(1.0, tf.float32), shape=[])

        self.learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[])

    def build_layers(self, vocabulary_size, embedding_dim, n_hidden_rnn, n_tags):
        """Specifies bi-LSTM architecture and computes logits for inputs."""

        # Create embedding variable (tf.Variable) with dtype tf.float32
        initial_embedding_matrix = np.random.randn(vocabulary_size, embedding_dim) / np.sqrt(embedding_dim)
        embedding_matrix_variable = tf.Variable(initial_embedding_matrix, name='embedding_matrix', dtype=tf.float32)

        # Create RNN cells (for example, tf.nn.rnn_cell.BasicLSTMCell) with n_hidden_rnn number of units
        # and dropout (tf.nn.rnn_cell.DropoutWrapper), initializing all *_keep_prob with dropout placeholder.
        forward_cell = tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden_rnn, forget_bias=3.0),
            input_keep_prob=self.dropout_ph,
            output_keep_prob=self.dropout_ph,
            state_keep_prob=self.dropout_ph
        )

        backward_cell = tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden_rnn, forget_bias=3.0),
            input_keep_prob=self.dropout_ph,
            output_keep_prob=self.dropout_ph,
            state_keep_prob=self.dropout_ph
        )

        # Look up embeddings for self.input_batch (tf.nn.embedding_lookup).
        # Shape: [batch_size, sequence_len, embedding_dim].
        embeddings = tf.nn.embedding_lookup(embedding_matrix_variable, self.input_batch)

        # Pass them through Bidirectional Dynamic RNN (tf.nn.bidirectional_dynamic_rnn).
        # Shape: [batch_size, sequence_len, 2 * n_hidden_rnn].
        # Also don't forget to initialize sequence_length as self.lengths and dtype as tf.float32.
        (rnn_output_fw, rnn_output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=forward_cell, cell_bw=backward_cell,
            dtype=tf.float32,
            inputs=embeddings,
            sequence_length=self.lengths
        )
        rnn_output = tf.concat([rnn_output_fw, rnn_output_bw], axis=2)

        # Dense layer on top.
        # Shape: [batch_size, sequence_len, n_tags].
        self.logits = tf.layers.dense(rnn_output, n_tags, activation=None)

    def compute_predictions(self):
        """Transforms logits to probabilities and finds the most probable tags."""

        # Create softmax (tf.nn.softmax) function
        softmax_output = tf.nn.softmax(self.logits)

        # Use argmax (tf.argmax) to get the most probable tags
        # Don't forget to set axis=-1
        # otherwise argmax will be calculated in a wrong way
        self.predictions = tf.argmax(softmax_output, axis=-1)

    def compute_loss(self, n_tags, PAD_index):
        """Computes masked cross-entopy loss with logits."""

        # Create cross entropy function function (tf.nn.softmax_cross_entropy_with_logits)
        ground_truth_tags_one_hot = tf.one_hot(self.ground_truth_tags, n_tags)
        loss_tensor = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_tags_one_hot, logits=self.logits)

        mask = tf.cast(tf.not_equal(self.input_batch, PAD_index), tf.float32)
        # Create loss function which doesn't operate with <PAD> tokens (tf.reduce_mean)
        # Be careful that the argument of tf.reduce_mean should be
        # multiplication of mask and loss_tensor.
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(loss_tensor, mask), axis=-1) / tf.reduce_sum(mask, axis=-1))

    def perform_optimization(self):
        """Specifies the optimizer and train_op for the model."""

        # Create an optimizer (tf.train.AdamOptimizer)
        self.optimizer =  tf.train.AdamOptimizer(self.learning_rate_ph)
        ######### YOUR CODE HERE #############
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)

        # Gradient clipping (tf.clip_by_norm) for self.grads_and_vars
        # Pay attention that you need to apply this operation only for gradients
        # because self.grads_and_vars contains also variables.
        # list comprehension might be useful in this case.
        clip_norm = tf.cast(1.0, tf.float32)
        self.grads_and_vars = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in self.grads_and_vars]
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)

    def __init__(self, vocabulary_size, n_tags, embedding_dim, n_hidden_rnn, PAD_index):
        self.declare_placeholders()
        self.build_layers(vocabulary_size, embedding_dim, n_hidden_rnn, n_tags)
        self.compute_predictions()
        self.compute_loss(n_tags, PAD_index)
        self.perform_optimization()

    def train_on_batch(self, session, x_batch, y_batch, lengths, learning_rate, dropout_keep_probability):
        feed_dict = {self.input_batch: x_batch,
                     self.ground_truth_tags: y_batch,
                     self.learning_rate_ph: learning_rate,
                     self.dropout_ph: dropout_keep_probability,
                     self.lengths: lengths}

        session.run(self.train_op, feed_dict=feed_dict)

    def predict_for_batch(self, session, x_batch, lengths):
        feed_dict = {
            self.input_batch: x_batch,
            self.lengths: lengths
        }
        predictions = session.run(self.predictions, feed_dict=feed_dict)
        return predictions


def predict_tags(model, session, token_idxs_batch, lengths):
    """Performs predictions and transforms indices to tokens and tags."""
    
    tag_idxs_batch = model.predict_for_batch(session, token_idxs_batch, lengths)
    
    tags_batch, tokens_batch = [], []
    for tag_idxs, token_idxs in zip(tag_idxs_batch, token_idxs_batch):
        tags, tokens = [], []
        for tag_idx, token_idx in zip(tag_idxs, token_idxs):
            tags.append(data.idx2tag[tag_idx])
            tokens.append(data.idx2token[token_idx])
        tags_batch.append(tags)
        tokens_batch.append(tokens)
    return tags_batch, tokens_batch
    
    
def eval_conll(model, session, tokens, tags, short_report=True):
    """Computes NER quality measures using CONLL shared task script."""
    
    y_true, y_pred = [], []
    for x_batch, y_batch, lengths in data.batches_generator(1, tokens, tags):
        tags_batch, tokens_batch = predict_tags(model, session, x_batch, lengths)
        if len(x_batch[0]) != len(tags_batch[0]):
            raise Exception("Incorrect length of prediction for the input, "
                            "expected length: %i, got: %i" % (len(x_batch[0]), len(tags_batch[0])))
        predicted_tags = []
        ground_truth_tags = []
        for gt_tag_idx, pred_tag, token in zip(y_batch[0], tags_batch[0], tokens_batch[0]): 
            if token != '<PAD>':
                ground_truth_tags.append(data.idx2tag[gt_tag_idx])
                predicted_tags.append(pred_tag)

        # We extend every prediction and ground truth sequence with 'O' tag
        # to indicate a possible end of entity.
        y_true.extend(ground_truth_tags + ['O'])
        y_pred.extend(predicted_tags + ['O'])
        
    results = precision_recall_f1(y_true, y_pred, print_results=True, short_report=short_report)
    return results


# ## Run your experiment

# Create *BiLSTMModel* model with the following parameters:
#  - *vocabulary_size* — number of tokens;
#  - *n_tags* — number of tags;
#  - *embedding_dim* — dimension of embeddings, recommended value: 200;
#  - *n_hidden_rnn* — size of hidden layers for RNN, recommended value: 200;
#  - *PAD_index* — an index of the padding token (`<PAD>`).
# 
# Set hyperparameters. You might want to start with the following recommended values:
# - *batch_size*: 32;
# - 4 epochs;
# - starting value of *learning_rate*: 0.005
# - *learning_rate_decay*: a square root of 2;
# - *dropout_keep_probability*: try several values: 0.1, 0.5, 0.9.
# 
# However, feel free to conduct more experiments to tune hyperparameters and earn extra points for the assignment.

# In[71]:


tf.reset_default_graph()

model = BiLSTMModel(
    vocabulary_size=data.vocab_size,
    n_tags=21, 
    embedding_dim=200, 
    n_hidden_rnn=200,
    PAD_index=data.token2idx['<PAD>']
)


batch_size = 32
n_epochs = 4
learning_rate = 0.005
learning_rate_decay = np.sqrt(2)
dropout_keep_probability = 0.5


sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('Start training... \n')
for epoch in range(n_epochs):
    # For each epoch evaluate the model on train and validation data
    print('-' * 20 + ' Epoch {} '.format(epoch+1) + 'of {} '.format(n_epochs) + '-' * 20)
    print('Train data evaluation:')
    eval_conll(model, sess, data.train_tokens, data.train_tags, short_report=True)
    print('Validation data evaluation:')
    eval_conll(model, sess, data.validation_tokens, data.validation_tags, short_report=True)
    
    # Train the model
    for x_batch, y_batch, lengths in data.batches_generator(batch_size, data.train_tokens, data.train_tags):
        model.train_on_batch(sess, x_batch, y_batch, lengths, learning_rate, dropout_keep_probability)
        
    # Decaying the learning rate
    learning_rate = learning_rate / learning_rate_decay
    print('learning rate is %s' % learning_rate)
    
print('...training finished.')


print('-' * 20 + ' Train set quality: ' + '-' * 20)
train_results = eval_conll(model, sess, data.train_tokens, data.train_tags, short_report=False)

print(train_results)

# print('-' * 20 + ' Validation set quality: ' + '-' * 20)
# validation_results = ######### YOUR CODE HERE #############
#
# print('-' * 20 + ' Test set quality: ' + '-' * 20)
# test_results = ######### YOUR CODE HERE #############

