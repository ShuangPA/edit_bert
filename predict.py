import optparse
import os
import sys
sys.path.append('../insight_nlp')
from pa_nlp.measure import Measure
import tensorflow as tf
from _data import Dataset
from _model import Model
import numpy as np

class Predictor:
  def __init__(self, bert_config_file, max_seq_length, num_labels):
    self._graph = tf.Graph()
    with self._graph.as_default():
      self._model = Model(bert_config_file, max_seq_length, None, False, num_labels)
    self._sess = tf.Session(graph=self._graph)

  def load_model(self, model_path):
    print(f"loading model from: {model_path}")
    with self._graph.as_default():
      tf.train.Saver().restore(
        self._sess,model_path
      )
  def predict_dataset(self, dataset):
    data_reader = dataset.get_data_reader(32,1)
    all_true_label = []
    all_pred_label = []
    all_pred_prob = []
    for batch_id, batch_data in data_reader.get_batch_data():
      [input_ids_b, input_mask_b, segment_ids_b, label_ids_b, is_real_example_b] = batch_data
      all_prob = self._sess.run(
        fetches=[
          self._model.probabilities
        ],
        feed_dict={
          self._model.model.input_ids_p: input_ids_b,
          self._model.model.input_mask_p: input_mask_b,
          self._model.model.segment_ids_p: segment_ids_b,
        }
      )
      pred_label = [np.argmax(np.array(item)) for item in all_prob[0]]
      pred_prob = [max(item) for item in all_prob[0]]
      all_true_label.extend(label_ids_b)
      all_pred_label.extend(pred_label)
      all_pred_prob.extend(pred_prob)

    _eval = Measure.calc_precision_recall_fvalue(all_true_label, all_pred_label)

    print(_eval)

  def predict_sentence(self, sen):
    pass

def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  parser.add_option("--gpu", default="-1", help="default=-1")
  parser.add_option("--bert_config_file", default="../bert_data/uncased_L-12_H-768_A-12/bert_config.json")
  parser.add_option("--num_labels", default=2)
  parser.add_option("--dev_file", default="../STS/data/std_quora_data/dev0.pydict")
  parser.add_option("--vocab_file", default="../bert_data/uncased_L-12_H-768_A-12/vocab.txt")
  parser.add_option("--temp_dir", default="./temp")
  parser.add_option("--max_seq_length", default=128)
  parser.add_option("--model_ckpt", default="./out_temp/model-200")
d
  (options, args) = parser.parse_args()
  print(options)
  os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu

  P = Predictor(
    bert_config_file=options.bert_config_file,
    max_seq_length=options.max_seq_length,
    num_labels=options.num_labels,
  )
  os.system(f'mkdir {options.temp_dir}')
  dev_data = Dataset(
    file_name=options.dev_file,
    num_class=options.num_labels,
    vocab_file=options.vocab_file,
    do_lower_case=True,
    tf_record_dir=options.temp_dir,
    if_train=False,
    max_seq_length=options.max_seq_length
  )

  P.load_model(options.model_ckpt)
  P.predict_dataset(dev_data)

if __name__ == '__main__':
  main()
