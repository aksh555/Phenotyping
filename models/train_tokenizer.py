from tokenizers import BertWordPieceTokenizer

# Initialize an empty BERT tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text=False,
    strip_accents=False,
    lowercase=True,
)

files = ['disch_rad_notes.txt']

# train BERT tokenizer
tokenizer.train(
    files,
    vocab_size=50000,
    min_frequency=3,
    show_progress=True,
    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
    limit_alphabet=1000,
    wordpieces_prefix="##"
)

# save the vocab
tokenizer.save_model('disch_rad_notes_tokenizer')


# # create a BERT tokenizer with trained vocab
# vocab = 'bert-vocab.txt'
# tokenizer = BertWordPieceTokenizer(vocab)

# # test the tokenizer with some text
# encoded = tokenizer.encode('...')
# print(encoded.tokens)
