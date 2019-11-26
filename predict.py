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
  def __init__(self, bert_config_file, num_labels):
    self._graph = tf.Graph()
    with self._graph.as_default():
      self._model = Model(bert_config_file, False, num_labels)
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
          self._model.input_ids_p: input_ids_b,
          self._model.input_mask_p: input_mask_b,
          self._model.segment_ids_p: segment_ids_b,
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
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--gpu", default="-1", help="default=-1")

  # default=False, help="")
  (options, args) = parser.parse_args()
  print(options)
  os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu

  P = Predictor(
    bert_config_file='../bert_data/uncased_L-12_H-768_A-12/bert_config.json',
    num_labels=2,
  )
  os.system('mkdir ./temp')
  dev_data = Dataset(
    file_name='../STS/data/std_quora_data/dev1.pydict',
    num_class=2,
    vocab_file='../bert_data/uncased_L-12_H-768_A-12/vocab.txt',
    do_lower_case=True,
    tf_record_dir='./temp',
    if_train=False,
    max_seq_length=128
  )

  P.load_model('./out01/model-150000')
  P.predict_dataset(dev_data)

if __name__ == '__main__':
  main()
