import open_source.tokenization as tokenization

class Tokenizer:
  def __init__(self, vocab_file, do_lower_case=True):
    self.vocab_file = vocab_file
    self.do_lower_case = do_lower_case
    self.tokenizer = tokenization.FullTokenizer(
      self.vocab_file, self.do_lower_case)

  def sentence_to_tokens(self, sentence):
    '''
    :param sentence: input English sentence
    :return: a list of tokens
    '''
    tokens = self.tokenizer.tokenize(sentence)
    return tokens

  def tokens_to_ids(self, tokens):
    '''
    :param tokens: a list of tokens
    :return: a list of ids corresponding to the tokens based on the vocab file
    '''
    ids = self.tokenizer.convert_tokens_to_ids(tokens)
    return ids

def main():
  T = Tokenizer(vocab_file='../../bert_data/uncased_L-12_H-768_A-12/vocab.txt')
  token = T.sentence_to_tokens('i am happy today 好的 好de')
  ids = T.tokens_to_ids(token)
  print(token)
  print(ids)

if __name__ == '__main__':
  main()