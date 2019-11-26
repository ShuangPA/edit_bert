import optparse
import os
import tensorflow as tf
import Bert.open_source.modeling as modeling
import optimization
import time
from _data import Dataset
from _model import Model
from predict import Predictor

class Trainer:
  def __init__(self, bert_config_file, is_training, num_labels,
               train_file, dev_file, vocab_file, output_dir, max_seq_length,
               learning_rate, batch_size, epochs, warmup_proportion,
               virtual_batch_size_ratio, evaluate_every, init_ckpt
               ):
    os.system(f"mkdir {output_dir}")
    self._data_train = Dataset(train_file, num_labels, vocab_file, True,
                               output_dir, True, max_seq_length)
    self._dev_data = Dataset(dev_file, num_labels, vocab_file, True,
                             output_dir, False, max_seq_length)
    num_train_step = int(self._data_train.size / batch_size * epochs)
    num_warmup_step = int(num_train_step * warmup_proportion)

    self._model = Model(bert_config_file, max_seq_length, init_ckpt, is_training, num_labels)

    self._train_op, self._global_step = optimization.create_optimizer(
      self._model.loss, learning_rate, num_train_step, num_warmup_step, False, virtual_batch_size_ratio)

    self.batch_size = batch_size
    self.epochs = epochs
    self.evaluate_every = evaluate_every
    self.output_dir = output_dir
    self._predictor = Predictor(bert_config_file, max_seq_length, num_labels)
    
  def evaluate(self, step):
    print(f"saving model[{step}] ...")
    self._saver.save(
      self._sess, os.path.join(f"{self.output_dir}/model"), global_step=step
    )
    self._predictor.load_model(f"{self.output_dir}/model-{step}")
    self._predictor.predict_dataset(self._dev_data)

  def train(self):
    self._sess = tf.Session()

    tvars = tf.trainable_variables()
    initialized_variable_names = self._model.model.initialized_variable_names
    print('*** loading variablrs ***')
    for var in tvars:
      if var.name in initialized_variable_names:
        print(f"name = {var.name}, shape = {var.shape}, *INIT_FROM_CKPT*")
      else:
        print(f"name = {var.name}, shape = {var.shape}")

    self._sess.run(tf.global_variables_initializer())
    self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

    data_reader = self._data_train.get_data_reader(self.batch_size, self.epochs)

    for batch_id, batch_data in data_reader.get_batch_data():
      [input_ids_b, input_mask_b, segment_ids_b, label_ids_b, is_real_example_b] = batch_data
      time_start = time.time()
      train_loss, global_step_train,  _ = self._sess.run(
        fetches=[
          self._model.loss,
          self._global_step,
          self._train_op
        ],
        feed_dict={
          self._model.model.input_ids_p: input_ids_b,
          self._model.model.input_mask_p: input_mask_b,
          self._model.model.segment_ids_p: segment_ids_b,
          self._model.labels: label_ids_b
        }
      )
      time_duration = time.time() - time_start
      print(f"batch: {batch_id}, global_step: {global_step_train}, "
            f"loss: {train_loss:.4f}, time: {time_duration}")
      if global_step_train != 0 and global_step_train % self.evaluate_every == 0:
        self.evaluate(global_step_train)


def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--gpu", default="-1", help="default=-1")

  # default=False, help="")
  (options, args) = parser.parse_args()
  print(options)
  os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu

  T = Trainer(
    bert_config_file = '../bert_data/uncased_L-12_H-768_A-12/bert_config.json',
    is_training = True,
    num_labels = 2,
    train_file = '../STS/data/std_quora_data/s.train.pydict',
    dev_file = '../STS/data/std_quora_data/s.train.pydict',
    vocab_file = '../bert_data/uncased_L-12_H-768_A-12/vocab.txt',
    output_dir = './out_temp',
    max_seq_length = 128,
    learning_rate = 5e-5,
    batch_size = 64,
    epochs = 30,
    warmup_proportion = 0.1,
    virtual_batch_size_ratio = 1,
    evaluate_every = 200,
    init_ckpt = '../bert_data/uncased_L-12_H-768_A-12/bert_model.ckpt'
  )
  T.train()

if __name__ == '__main__':
  main()