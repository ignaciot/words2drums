import re
import torch
from transformers import BertTokenizer, BertModel
from numpy import mean, std, load, asarray, array, percentile
from keras.models import load_model
from transformers import logging
logging.set_verbosity_error()


n_outputs = 129
track_length = 32

test_phrases = []

with open("inputs.txt", 'r') as fd:
    for line in fd.readlines():
        test_phrases.append(line.replace('\n', ''))

bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model.eval()
test_embeddings = []

for phrase in test_phrases:
    print("Processing mood terms for test words: \"%s\"" % phrase)
    marked_text = "[CLS] " + phrase + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_ids = [1] * len(tokenized_text)
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        outputs = bert_model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
    token_vecs = hidden_states[-2][0]
    mood_embedding = array(torch.mean(token_vecs, dim=0))
    test_embeddings.append(mood_embedding)

model = load_model("trained_words2drums")


for word_idx in range(len(test_phrases)):
    yhat = model.predict(asarray([test_embeddings[word_idx]]))
    pred_q3 = percentile(yhat[0][1:], 75, method = 'midpoint')
    print('\nPrediction for "%s": Tempo=%d bpm' % (test_phrases[word_idx], yhat[0][0]))
    drum_track_str = ''.join(['1' if x > pred_q3 else '0' for x in yhat[0][1:]])
    online_seq_str = "Online Sequencer:319887:"
    idx = 1
    online_sequencer_drum_type = ['D4', 'C3', 'D5', 'F#3']
    drum_type = online_sequencer_drum_type.pop()
    tempo = str(int(yhat[0][0]))
    track_str = 'Suggested tempo: ' + tempo + '\n'
    while idx + track_length <= n_outputs:
        track_info = yhat[0][idx:idx+track_length]
        track_str += '\t|' + re.sub("(.{16})", "\\1|", ''.join(['X' if x > pred_q3 else '-' for x in track_info]), 0, re.DOTALL) + '\n'
        track_idx = 0
        for beat in track_info:
            if beat > pred_q3:
                online_seq_str += str(track_idx) + " " + drum_type + " 1 2;"
            track_idx += 1
        if len(online_sequencer_drum_type) > 0:
            drum_type = online_sequencer_drum_type.pop()
        idx = idx+track_length
    print(online_seq_str + ":")
    print(track_str)

