from ast import literal_eval

import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizerFast


def split_sentence(tokenizer, text, spans=None, max_sentence_length=500):
    encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
    sentences_offsets = [offset[1] for input_ids, offset in zip(encoded['input_ids'], encoded['offset_mapping'])
                         if input_ids == 1012]
    if len(sentences_offsets) == 0 or len(text) - sentences_offsets[-1] > 0:
        sentences_offsets.append(len(text))

    new_split = [0]
    for split_offset_id in range(len(sentences_offsets) - 1):
        if sentences_offsets[split_offset_id + 1] - new_split[-1] > max_sentence_length:
            new_split.append(sentences_offsets[split_offset_id])

    if not new_split[-1] == sentences_offsets[-1]:
        new_split.append(sentences_offsets[-1])

    if spans:
        spans_vector = [i in row['spans'] for i in range(len(row['text']))]

    texts, new_spans, offsets = [], [], []
    for offset_id in range(len(new_split) - 1):
        a, b = new_split[offset_id], new_split[offset_id + 1]
        texts.append(text[a:b])
        offsets.append(new_split[offset_id])

        if spans:
            spans_in_subtext = [idx for idx, is_true in enumerate(spans_vector[a:b]) if is_true]
            new_spans.append(spans_in_subtext)

    if not spans:
        new_spans = [[] for _ in range(len(offsets))]

    return texts, offsets, new_spans


if __name__ == '__main__':
    df = pd.read_csv('data/spans/tsd_train.csv')
    df.loc[:, 'spans'] = df['spans'].apply(literal_eval)
    pd.set_option('display.max_columns', 500)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)

    global_texts, global_spans, global_id, global_offset = [], [], [], []

    for row_id, row in tqdm(df.iterrows(), total=len(df)):
        texts, offsets, spans = split_sentence(tokenizer, row['text'], row['spans'])

        ids = list([row_id for i in range(len(offsets))])
        global_texts.extend(texts)
        global_spans.extend(spans)
        global_id.extend(ids)
        global_offset.extend(offsets)

        recovered_text = ''.join(texts)
        recovered_spans = [span_start + offset for span_list, offset in zip(spans, offsets) for span_start in span_list]

        assert recovered_text == row['text']
        assert recovered_spans == row['spans']
        assert len(ids) == len(texts) == len(offsets) == len(spans)

    new_df = pd.DataFrame({
        'text': global_texts,
        'spans': global_spans,
        'sentence_id': global_id,
        'offset': global_offset
    })

    print(new_df.head())
    new_df.to_csv('data/spans/tsd_train_500.csv', index=False)
