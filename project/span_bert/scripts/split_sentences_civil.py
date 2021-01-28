import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizerFast

from scripts.split_sentences import split_sentence

if __name__ == '__main__':
    df = pd.read_csv('data/civil_comments/all_data.csv')
    df = df[df['toxicity'] > 0.5]
    print(len(df))

    pd.set_option('display.max_columns', 500)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)

    global_texts, global_id, global_offset = [], [], []

    for row_id, row in tqdm(df.iterrows(), total=len(df)):
        texts, offsets, _ = split_sentence(tokenizer, row['comment_text'], max_sentence_length=500)

        ids = list([row_id for i in range(len(offsets))])
        global_texts.extend(texts)
        global_id.extend(ids)
        global_offset.extend(offsets)

        recovered_text = ''.join(texts)

        assert recovered_text == row['comment_text']
        assert len(ids) == len(texts) == len(offsets)

    new_df = pd.DataFrame({
        'text': global_texts,
        'sentence_id': global_id,
        'offset': global_offset
    })

    print(new_df.head())
    new_df.to_csv('data/civil_comments/all_civil_data_512.csv', index=False)
