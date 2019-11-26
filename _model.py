import tensorflow as tf
import modeling

class Model:
  def __init__(self, bert_config_file, is_training, num_labels):
    self.input_ids_p = tf.placeholder(tf.int32, [None, 128], name="input_ids")
    self.input_mask_p = tf.placeholder(tf.int32, [None, 128], name="input_mask")
    self.segment_ids_p = tf.placeholder(tf.int32, [None, 128],name="segment_ids")
    self.labels = tf.placeholder(tf.int32, [None], name='label_ids')
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    self.model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=self.input_ids_p,
      input_mask=self.input_mask_p,
      token_type_ids=self.segment_ids_p
    )
    output_layer = self.model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value
    output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())
    with tf.variable_scope("loss"):
      if is_training:
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

      self.logits = tf.matmul(output_layer, output_weights, transpose_b=True)
      self.logits = tf.nn.bias_add(self.logits, output_bias)
      self.probabilities = tf.nn.softmax(self.logits, axis=-1)
      self.log_probs = tf.nn.log_softmax(self.logits, axis=-1)

      self.one_hot_labels = tf.one_hot(self.labels, depth=num_labels,
                                       dtype=tf.float32)

      self.per_example_loss = -tf.reduce_sum(
        self.one_hot_labels * self.log_probs, axis=-1)
      self.loss = tf.reduce_mean(self.per_example_loss)