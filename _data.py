import os
import sys
sys.path.append('../insight_nlp')
from pa_nlp.tf_1x import *
from pa_nlp.tf_1x import nlp_tf
from pa_nlp.tf_1x.estimator.dataset import DataReaderBase
from pa_nlp.tf_1x.estimator.param import ParamBase
import tensorflow as tf
import collections
import Bert.open_source.tokenization as tokenization

tf.logging.set_verbosity(tf.logging.INFO)

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label

class Dataset:
  def __init__(self, file_name, num_class, vocab_file, do_lower_case,
               tf_record_dir, if_train, max_seq_length):
    label_list = [str(i) for i in range(num_class)]
    tokenizer = tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)
    if if_train:
      filename = os.path.join(tf_record_dir, 'train.tf_record')
      data_example = self._create_examples(file_name, 'train')
      self.size = len(data_example)
    else:
      filename = os.path.join(tf_record_dir, 'eval.tf_record')
      data_example = self._create_examples(file_name, 'dev')
      self.size = len(data_example)
    self.file_based_convert_examples_to_features(
        data_example, label_list, max_seq_length, tokenizer, filename)

    self.max_seq_length = max_seq_length
    self.filename = filename

  def get_data_reader(self, batch_size, epochs):
    seq_length = self.max_seq_length
    tf_file = self.filename
    class MyDataReader(DataReaderBase):
      def parse_example(self, serialized_example):
        data_fields = {
          "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
          "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
          "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
          "label_ids": tf.FixedLenFeature([], tf.int64),
          "is_real_example": tf.FixedLenFeature([], tf.int64),
        }
        parsed = tf.parse_single_example(serialized_example, data_fields)

        input_ids = parsed["input_ids"]
        input_mask = parsed["input_mask"]
        segment_ids = parsed["segment_ids"]
        label_ids = parsed["label_ids"]
        is_real_example = parsed["is_real_example"]

        return input_ids, input_mask, segment_ids, label_ids, is_real_example

    param = ParamBase("debug_model")
    param.batch_size = batch_size
    param.epoch_num = epochs
    data_reader = MyDataReader(tf_file, param, True)

    return data_reader

  def _create_examples(self, filename, set_type):
    examples = []
    f = open(filename, 'r').readlines()
    for i, line in enumerate(f):
      info = eval(line)
      text_a = info['sen1']
      text_b = info['sen2']
      label = str(info['class'])
      guid = "%s-%s" % (set_type, i)
      examples.append(
        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def file_based_convert_examples_to_features(
    self, examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
      if ex_index % 10000 == 0:
        tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

      feature = self.convert_single_example(ex_index, example, label_list,
                                       max_seq_length, tokenizer)

      def create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

      features = collections.OrderedDict()
      features["input_ids"] = create_int_feature(feature.input_ids)
      features["input_mask"] = create_int_feature(feature.input_mask)
      features["segment_ids"] = create_int_feature(feature.segment_ids)
      features["label_ids"] = create_int_feature([feature.label_id])
      features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

      tf_example = tf.train.Example(
        features=tf.train.Features(feature=features))
      writer.write(tf_example.SerializeToString())
    writer.close()

  def convert_single_example(self, ex_index, example, label_list,
                             max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
      return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
      label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 5:
      tf.logging.info("*** Example ***")
      tf.logging.info("guid: %s" % (example.guid))
      tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
      tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      tf.logging.info(
        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
      tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
    return feature

  def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
      total_length = len(tokens_a) + len(tokens_b)
      if total_length <= max_length:
        break
      if len(tokens_a) > len(tokens_b):
        tokens_a.pop()
      else:
        tokens_b.pop()



class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example
