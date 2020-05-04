#import fire
import os
import json
from itertools import groupby
from razdel import sentenize, tokenize
from collections import namedtuple
from pathlib import Path
#from transformers.tokenization_bert import BertTokenizer
import sys
import pandas as pd
from tqdm import tqdm
import numpy as np

#sys.path.append('../task6_baseline')
#from data_utils import multi_predictions_postprocessing

SpanText = namedtuple('SpanText', ['start', 'stop', 'text'])


def sentenize_wrap(text):
    return [
        SpanText(start=sentence.start, stop=sentence.stop, text=sentence.text) for sentence in sentenize(text)
    ]


def tokenize_wrap(sentence):
    return [
        SpanText(start=token.start, stop=token.stop, text=token.text) for token in tokenize(sentence)
    ]


def get_tokenized_dataset(rurebus_dir: str):
    annotations = [file.rstrip('tx') for file in os.listdir(rurebus_dir) if file.endswith('.txt')]
    result = {}
    for annotation_file in annotations:
        file_txt = open(
            os.path.join(rurebus_dir, annotation_file) + 'txt').read()
        file_sentences = [sentence for sentence in sentenize_wrap(file_txt)]
        file_tokenized_sentences = [[tok for tok in tokenize_wrap(sentence.text)] for sentence in file_sentences]

        result[annotation_file.rstrip('.')] = {
            'file_sentences': (file_sentences, file_tokenized_sentences)
        }
    return result


def get_annotated_dataset(rurebus_dir: str):
    annotations = [file.rstrip('an') for file in os.listdir(rurebus_dir) if file.endswith('.ann')]
    result = {}
    for annotation_file in annotations:
        file_ann = [line.strip() for line in open(
            (os.path.join(rurebus_dir, annotation_file) + 'ann'), encoding = 'utf-8').readlines()]
        file_ann = [tuple(line.split('\t')) for line in file_ann]
        entities = [line for line in file_ann if line[0].startswith('T')]
        ignored_entities = set([line[0] for line in entities if len(line[1].split()) != 3])
        entities = [(line[0], line[1], '\t'.join(line[2:])) for line in entities if
                    line[0] not in ignored_entities]
        try:
            entities_map = dict(
                [(tag, (attrs.split(' ')[0], tuple([int(position) for position in attrs.split(' ')[1:]]), tag_text)) for
                 (tag, attrs, tag_text) in entities])
        except ValueError:
            print(annotation_file)
            entities_map = dict(
                [(tag, (attrs.split(' ')[0], tuple(
                    [int(position) if ';' not in position else int(position.split(';')[0]) for position in
                     attrs.split(' ')[1:]]), tag_text)) for
                 (tag, attrs, tag_text) in entities])
            # raise ValueError

        relations = [line for line in file_ann if line[0].startswith('R')]
        r_args = [line[1].split(' ')[1][5:] for line in relations]
        l_args = [line[1].split(' ')[2][5:] for line in relations]
        relations = [line for l_arg, r_arg, line in zip(r_args, l_args, relations) if
                     r_arg not in ignored_entities or r_arg not in ignored_entities]
        relations_map = dict([
            (
                tag,
                tuple([attr if attr_id == 0 else attr.split(':')[-1] for attr_id, attr in enumerate(attrs.split(' '))]))
            for (tag, attrs) in relations
        ])

        new_relation_map = {}
        for tag, (relation_type, subj, obj) in relations_map.items():
            if subj not in new_relation_map:
                new_relation_map[subj] = set()
            new_relation_map[subj].add((obj, relation_type))

        result[annotation_file.rstrip('.')] = {
            'entities_map': entities_map,
            'relations_map': relations_map,
            'subj_to_obj_map': new_relation_map
        }

    return result


def filter_entities_dict_by_text_span(entities_dict, span_start, span_end):
    entities_from_span = [
        tag for tag, (_, (start, end), _) in entities_dict['entities_map'].items() if
        span_start <= start <= end <= span_end
    ]
    result = {
        'entities_map': dict([(tag, entities_dict['entities_map'][tag]) for tag in entities_from_span])
    }
    return result


def get_entity_tag_from_token_span(entities_dict, token_start, token_end):
    result = [
        (tag, entity_tag) for tag, (entity_tag, (start, end), _) in entities_dict['entities_map'].items() if
        start <= token_start <= token_end <= end
    ]
    if len(result) > 0:
        result = result[0]
    else:
        result = ('O', 'O')
    return result


def get_tag_and_relation_dataset(rurebus_data_dir: str, part: str = 'train_part_3'):
    rurebus_dir = os.path.join(rurebus_data_dir, part)
    annotatated_dataset = get_annotated_dataset(rurebus_dir=rurebus_dir)
    tokenized_dataset = get_tokenized_dataset(rurebus_dir=rurebus_dir)

    examples = []
    for annotated_file in tokenized_dataset:
        sentences, tokenized_sentences = tokenized_dataset[annotated_file]['file_sentences']
        for sent_id, (sentence, tokenized_sentence) in enumerate(zip(sentences, tokenized_sentences)):
            entities_from_span = filter_entities_dict_by_text_span(
                annotatated_dataset[annotated_file],
                sentence.start, sentence.stop
            )
            example_tags, tag_marks = [], []
            finded_entities = set()
            prev_tag = 'O'
            for token in tokenized_sentence:
                tag, entity_tag = get_entity_tag_from_token_span(entities_from_span, sentence.start + token.start,
                                                                 sentence.start + token.stop)
                if prev_tag == tag:
                    entity_tag = 'I-' + entity_tag if entity_tag != 'O' else 'O'
                else:
                    prev_tag = tag
                    entity_tag = 'B-' + entity_tag if entity_tag != 'O' else 'O'

                example_tags.append(
                    entity_tag
                )
                tag_marks.append(tag)
                if tag.startswith('T'):
                    finded_entities.add(tag)

            token = [token.text for token in tokenized_sentence]
            subj_to_obj_rel = annotatated_dataset[annotated_file]['subj_to_obj_map']
            for relation_id, subj in enumerate(finded_entities):
                subj_ids = [i for i, tag in enumerate(tag_marks) if tag == subj]
                if subj in subj_to_obj_rel:
                    objs = subj_to_obj_rel[subj]
                else:
                    objs = set()
                relations = ['0'] * len(token)
                for (obj, relation) in objs:
                    obj_ids = [i for i, tag in enumerate(tag_marks) if tag == obj]
                    for i in obj_ids:
                        relations[i] = relation

                subj_start, subj_end = subj_ids[0], subj_ids[-1]
                example = {
                    'sent_type': 1,
                    'sent_start': -1,
                    'sent_end': -1,
                    'token': token,
                    'tag': example_tags,
                    'id': f'{part}-{annotated_file}-{sent_id}-{subj}',
                    # 'id': f'{part}-{annotated_file}-{sent_id}-{"_".join([str(subj_start), str(subj_end)])}',
                    'sentence': sentence,
                    'tokenized_sentence': tokenized_sentence,
                    'subj_start': subj_start,
                    'subj_end': subj_end,
                    'relation': relations
                }
                examples.append(example)

    return examples


def get_tag_dataset(rurebus_data_dir: str, part: str = 'train_part_3'):
    rurebus_dir = os.path.join(rurebus_data_dir, part)
    annotatated_dataset = get_annotated_dataset(rurebus_dir=rurebus_dir)
    tokenized_dataset = get_tokenized_dataset(rurebus_dir=rurebus_dir)

    dataset = tokenized_dataset if len(tokenized_dataset) < 0 else annotatated_dataset
    examples = []
    for annotated_file in dataset:
        sentences, tokenized_sentences = tokenized_dataset[annotated_file]['file_sentences']
        for sent_id, (sentence, tokenized_sentence) in enumerate(zip(sentences, tokenized_sentences)):
            entities_from_span = filter_entities_dict_by_text_span(
                annotatated_dataset[annotated_file],
                sentence.start, sentence.stop
            )
            example_tags = []
            prev_tag = 'O'
            for token in tokenized_sentence:
                tag, entity_tag = get_entity_tag_from_token_span(entities_from_span, sentence.start + token.start,
                                                                 sentence.start + token.stop)
                if prev_tag == tag:
                    entity_tag = 'I-' + entity_tag if entity_tag != 'O' else 'O'
                else:
                    prev_tag = tag
                    entity_tag = 'B-' + entity_tag if entity_tag != 'O' else 'O'

                example_tags.append(
                    entity_tag
                )

            token = [token.text for token in tokenized_sentence]

            example = {
                'token': token,
                'tag': example_tags,
                'id': f'{part}-{annotated_file}-{sent_id}',
                'sentence': sentence,
                'tokenized_sentence': tokenized_sentence
            }
            examples.append(example)

    return examples


def get_tag_test_dataset(rurebus_data_dir: str):
    tokenized_dataset = get_tokenized_dataset(rurebus_dir=rurebus_data_dir)
    examples = []
    for annotated_file in tokenized_dataset:
        sentences, tokenized_sentences = tokenized_dataset[annotated_file]['file_sentences']
        for sent_id, (sentence, tokenized_sentence) in enumerate(zip(sentences, tokenized_sentences)):
            example_tags = ['O'] * len(tokenized_sentence)
            token = [token.text for token in tokenized_sentence]
            example = {
                'token': token,
                'tag': example_tags,
                'id': f'test_part-{annotated_file}-{sent_id}',
                'sentence': sentence,
                'tokenized_sentence': tokenized_sentence
            }
            examples.append(example)
    return examples


def read_examples(examples_path):
    examples = json.load(open(examples_path))
    for i in range(len(examples)):
        examples[i]['sentence'] = SpanText(start=examples[i]['sentence'][0],
                                           stop=examples[i]['sentence'][1],
                                           text=examples[i]['sentence'][2])
        examples[i]['tokenized_sentence'] = [
            SpanText(start=token[0], stop=token[1], text=token[2]) for token in examples[i]['tokenized_sentence']
        ]
    return examples


def check_output_dir(output_dir: str, force_write: bool):
    if os.path.exists(output_dir):
        if force_write:
            print('overwriting existing files')
            # os.system(f'rm {os.path.join(output_dir, "*")}')
        else:
            raise RuntimeError
    else:
        os.makedirs(output_dir, exist_ok=True)


def write_tag_predictions_to_folder(output_dir: str, eval_examples_path, eval_predictions_path,
                                    force_write=False):
    check_output_dir(output_dir, force_write)

    eval_examples = read_examples(eval_examples_path)
    df = pd.read_csv(eval_predictions_path, sep='\t')

    columns_to_split = ['token', 'tag_label', 'tag_pred', 'scores']
    for column in columns_to_split:
        if 'scores' in column:
            df.loc[:, column] = df[column].apply(lambda x: [float(y) for y in x.split(' ')])
        else:
            df.loc[:, column] = df[column].apply(lambda x: x.split(' '))

    df.loc[:, 'source_file_id'] = df.id.apply(lambda x: '-'.join(x.split('-')[2:-1]))  # 1:-1 - unique .ann path
    df.loc[:, 'source_file_name'] = df.id.apply(lambda x: '-'.join(x.split('-')[2:-1]))
    df.loc[:, 'sent_id_in_source_file'] = df.id.apply(lambda x: '-'.join(x.split('-')[2:]))
    unique_source_file_ids = df.source_file_id.unique()

    for source_file_id in unique_source_file_ids:
        predicted_tags_and_examples_tuples = []
        tmp_df = df[df.source_file_id == source_file_id]
        tmp_examples = [example for example in eval_examples if
                        '-'.join(example['id'].split('-')[1:-1]) == source_file_id]
        assert len(tmp_df.source_file_name.unique()) == 1
        source_file_name = tmp_df.source_file_name.unique()[0]
        for row in tmp_df.itertuples():
            sent_id = row.sent_id_in_source_file

            corresponding_example = [example for example in tmp_examples if example['id'] == f'test_part-{sent_id}'][0]
            predicted_tags_and_examples_tuples.append((row.tag_pred, corresponding_example))

        write_tag_ann_to_file(os.path.join(output_dir, f"{source_file_name}.ann"), predicted_tags_and_examples_tuples)


def get_bio_spans(tags):
    start_ids = [i for i, tag in enumerate(tags) if tag.startswith('B-')]
    additional_start_ids = []
    prev_tag = 'O'
    for i, tag in enumerate(tags):
        if tag.startswith('I-') and prev_tag == 'O':
            additional_start_ids.append(i)
        prev_tag = tag
    start_ids += additional_start_ids

    stop_ids = []
    for start_id in start_ids:
        stop_id = start_id + 1
        while stop_id < len(tags) and tags[stop_id].startswith('I-'):
            stop_id += 1
        stop_ids.append(stop_id - 1)
    spans = [(start_id, stop_id) for start_id, stop_id in zip(start_ids, stop_ids)]
    return spans


def write_tag_ann_to_file(filename, predicted_tags_and_examples_tuples):
    encountered_tags = {}
    result = ''
    for example_id, (tags, example) in enumerate(predicted_tags_and_examples_tuples):
        sentence = example['sentence']
        offset = sentence.start
        tags += ['O'] * (len(example['token']) - len(tags))
        spans = get_bio_spans(tags)
        tokens = example['tokenized_sentence']
        for span_id, (token_start, token_stop) in enumerate(spans):
            tag = tags[token_start][2:]
            key = f'{example_id}-{span_id}'
            encountered_tags[key] = f'{len(encountered_tags) + 1}'
            start_token = tokens[token_start]
            stop_token = tokens[token_stop]
            entity_text = sentence.text[start_token.start:stop_token.stop]
            start = start_token.start + offset
            stop = stop_token.stop + offset
            result += f'T{encountered_tags[key]}\t{tag} {start} {stop}\t{entity_text}\n'

    print(result, file=open(filename, 'w'), end='')


def get_formatted_relations(relations, example, corresponding_examples):
    found_relations = []
    arg2 = {}
    arg1 = 'Arg1:' + example['id'].split('-')[-1]

    for ex in corresponding_examples:
        if ex['id'] == example['id']:
            continue
        key = ex['id'].split('-')[-1]
        for i, relation in enumerate(relations):
            if relation not in ['0', 'O'] and ex['subj_start'] <= i <= ex['subj_end']:
                if key not in arg2:
                    arg2[key] = set()
                arg2[key].add(relation)
    for key in arg2:
        for relation in arg2[key]:
            found_relations.append((relation, arg1, f'Arg2:{key}'))

    return found_relations


def write_relation_ann_to_file(filename, predicted_relations_and_examples_tuples):
    result = ''
    relation_id = 1
    for _, (relations, example, examples) in enumerate(predicted_relations_and_examples_tuples):
        formatted_relations = get_formatted_relations(relations, example, examples)
        for _, (relation, arg1, arg2) in enumerate(formatted_relations):
            result += f'R{relation_id}\t{relation} {arg1} {arg2}\n'
            relation_id += 1

    print(result, file=open(filename, 'w'), end='')


def aggregate_tags_by_max_score(mini_dataset):
    df = mini_dataset.copy().reset_index(drop=True)
    scores = np.array([x for x in df.tag_scores.values])
    tags = np.array([x for x in df.tag_pred.values])
    max_ids = np.argsort(scores, axis=0)[-1]
    aggregated_tags = []
    for column, row in enumerate(max_ids):
        aggregated_tags.append(tags[row, column])
    return aggregated_tags


def write_tag_and_relation_predictions_to_folder(output_dir: str, eval_examples_path, eval_predictions_path,
                                                 force_write=False, postprocess=True, prefix_part='test_part',
                                                 bert_tokenizer='bert-base-multilingual-uncased', offset_in_id=2):
    check_output_dir(os.path.join(output_dir, 'relations'), force_write)
    check_output_dir(os.path.join(output_dir, 'set_1'), force_write)

    eval_examples = read_examples(eval_examples_path)
    df = pd.read_csv(eval_predictions_path, sep='\t')

    columns_to_transform = [
        'TOKEN', 'tag_label', 'tag_pred', 'tag_scores', 'relation_label', 'relation_pred', 'relation_scores'
    ]
    if postprocess:
        bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer)
        df = multi_predictions_postprocessing(df, bert_tokenizer)
        df_copy = df.copy()
        for column in columns_to_transform:
            if 'scores' in column:
                df_copy.loc[:, column] = df_copy[column].apply(lambda x: ' '.join([str(y) for y in x]))
            else:
                df_copy.loc[:, column] = df_copy[column].apply(lambda x: ' '.join(x))
        df_copy.to_csv(eval_predictions_path + '_postprocessed.csv', sep='\t', index=False)
    else:
        for column in columns_to_transform:
            if 'scores' in column:
                df.loc[:, column] = df[column].apply(lambda x: [float(y) for y in x.split(' ')])
            else:
                df.loc[:, column] = df[column].apply(lambda x: x.split(' '))

    df.loc[:, 'source_file_name'] = df.id.apply(lambda x: x.split('-')[offset_in_id])
    df.loc[:, 'sent_id_in_source_file'] = df.id.apply(lambda x: x.split('-')[offset_in_id + 1])
    df.loc[:, 'subj_id_in_sentence'] = df.id.apply(lambda x: x.split('-')[offset_in_id + 2])
    unique_source_file_names = df.source_file_name.unique()

    for source_file_name in tqdm(unique_source_file_names, total=len(unique_source_file_names)):
        predicted_relations_and_examples_tuples = []
        predicted_tags_and_examples_tuples = []
        tmp_df = df[df.source_file_name == source_file_name]
        tmp_examples = [example for example in eval_examples if
                        example['id'].split('-')[1] == source_file_name]
        assert len(tmp_df.source_file_name.unique()) == 1
        source_file_name = tmp_df.source_file_name.unique()[0]

        unique_sent_ids_in_source_file = tmp_df.sent_id_in_source_file.unique()
        for sent_id_in_source_file in unique_sent_ids_in_source_file:
            sent_df = tmp_df[tmp_df.sent_id_in_source_file == sent_id_in_source_file]

            aggregated_tags = aggregate_tags_by_max_score(sent_df)
            corresponding_examples = [example for example in tmp_examples if
                                      example['id'].startswith(
                                          f'{prefix_part}-{source_file_name}-{sent_id_in_source_file}')]
            if not corresponding_examples:
                print(corresponding_examples, sent_df)
            predicted_tags_and_examples_tuples.append((aggregated_tags, corresponding_examples[0]))
            for row in sent_df.itertuples():
                subj_id = row.subj_id_in_sentence
                # print(subj_id, [example['id'] for example in corresponding_examples])
                corresponding_example = [example for example in corresponding_examples if
                                         example['id'].startswith(
                                             f'{prefix_part}-{source_file_name}-{sent_id_in_source_file}-{subj_id}')][0]
                relations = row.relation_pred
                if set(relations) == {'0'}:
                    continue
                else:
                    predicted_relations_and_examples_tuples.append(
                        (relations, corresponding_example, corresponding_examples))

        write_tag_ann_to_file(os.path.join(output_dir, 'set_1', f"{source_file_name}.ann"),
                              predicted_tags_and_examples_tuples)
        write_relation_ann_to_file(os.path.join(output_dir, 'relations', f"{source_file_name}.ann"),
                                   predicted_relations_and_examples_tuples)