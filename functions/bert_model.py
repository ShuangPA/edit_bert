import os
current_path = os.path.abspath(__file__)
father_path = os.path.dirname(os.path.dirname(current_path))
import sys
sys.path.append(father_path)
import tensorflow as tf
import modeling

class BertModel:
  def __init__(self,
               bert_config,
               max_seq_length,
               bert_init_ckpt=None,
               is_training=True):
    self.input_ids_p = tf.placeholder(tf.int32, [None, max_seq_length],
                                      name="input_ids")
    # input_ids_p refers to input_ids returned by BertTokenizer
    self.input_mask_p = tf.placeholder(tf.int32, [None, max_seq_length],
                                       name="input_mask")
    # input_mask_p refers to input_mask returned by BertTokenizer
    self.segment_ids_p = tf.placeholder(tf.int32, [None, max_seq_length],
                                       name="segment_ids")
    # input_mask_p refers to segment_ids returned by BertTokenizer
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    self.model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=self.input_ids_p,
      input_mask=self.input_mask_p,
      token_type_ids=self.segment_ids_p
    )
    if bert_init_ckpt:
      tvars = tf.trainable_variables()
      (assignment_map, initialized_variable_names
       ) = modeling.get_assignment_map_from_checkpoint(tvars, bert_init_ckpt)
      tf.train.init_from_checkpoint(bert_init_ckpt, assignment_map)

  def get_sequence_output_of_a_layer(self, layer_num=-1):
    """
    Gets final hidden layer of encoder.
    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.model.get_all_encoder_layers()[layer_num]
