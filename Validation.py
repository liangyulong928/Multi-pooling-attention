import csv
import os

import torch
from transformers import BertTokenizer
from MultiPoolingAttentionQA import BertMultiPoolingAttentionForQuestionAnswering
from Utils import convert_file_to_json, token_location_redirection, get_event_schema

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizer.from_pretrained("chinese-bert-wwm-ext")

source_data_path = './data/origin/DuEE1.0/'
encode_data_path = './data/process/DuEE1.0/'
save_model_path = './output/model/MPA/'


def valid(few_num):
    model = BertMultiPoolingAttentionForQuestionAnswering.from_pretrained("chinese-bert-wwm-ext")
    event_schema = get_event_schema(source_data_path, encode_data_path)


    if os.path.isfile(save_model_path + f"model_MultiPoolingAttention_few_{few_num}.pth"):
        model.load_state_dict(torch.load(save_model_path + f"model_MultiPoolingAttention_few_{few_num}.pth",
                                         map_location=torch.device('cpu')), False)
    else:
        raise FileNotFoundError("Missing training model")
    print('------------------------model evaluation----------------------')
    json_list = convert_file_to_json(source_data_path + f"duee_dev.json/duee_dev.json")
    data = []
    acc_num = 0
    predict_num = 0
    golden_num = 0
    for jsonl in json_list:
        for event in jsonl['event_list']:
            if event["event_type"] in event_schema:
                event_type, trigger, trigger_index = event['event_type'], event['trigger'], event['trigger_start_index']
                sentence_pro = jsonl['text'][:trigger_index] + '[SEP]' + \
                            jsonl['text'][trigger_index: trigger_index + len(trigger)] + '[SEP]' + \
                            jsonl['text'][trigger_index + len(trigger):]
                argument_index = {}
                for argument in event['arguments']:
                    argument_index[argument['role']] = argument['argument']
                arguments = event_schema[event['event_type']]
                for argument in arguments:
                    question = argument
                    context = question + '[SEP]' + sentence_pro
                    token = tokenizer.encode_plus(context[:254], max_length=256, padding='max_length', return_tensors='pt')
                    trigger = '[SEP]' + trigger + '[SEP]'
                    token_list = tokenizer.tokenize(context)
                    trigger_index_start, trigger_index_end = token_location_redirection(token_list, trigger, tokenizer)
                    input_token = {'input_ids': token['input_ids'],
                                'token_type_ids': token['token_type_ids'],
                                'attention_mask': token['attention_mask'],
                                'trigger_start_positions': torch.tensor([[trigger_index_start + 1]]),
                                'trigger_end_positions': torch.tensor([[trigger_index_end + 1]])}
                    output = model(**input_token)
                    predict_start = torch.argmax(output['start_logits']).item() - 1
                    predict_end = torch.argmax(output['end_logits']).item() - 1
                    golden_arg = argument_index.get(argument, None)
                    if predict_end >= predict_start >= 0:
                        predict_num += 1
                    if golden_arg:
                        golden_num += 1
                        answer_index_start, answer_index_end = token_location_redirection(token_list, golden_arg, tokenizer)
                        if predict_start <= answer_index_start <= predict_end:
                            acc_num += 1
                        row = [context,
                            golden_arg,
                            ''.join(token_list[predict_start: predict_end]),
                            answer_index_start, predict_start, predict_end]
                    else:
                        row = [context,
                            None,
                            ''.join(token_list[predict_start: predict_end]),
                            -1, predict_start, predict_end]
                    data.append(row)
    precision = acc_num / predict_num
    recall = acc_num / golden_num
    print(f"for fewer number is {few_num}:")
    print(f'precision = {precision} , recall = {recall} , f1 = {2 * (precision * recall) / (precision + recall)}')
    with open(save_model_path + f'MPA_eval_few_{few_num}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


if __name__ == '__main__':
    for fewer_num in [1, 2, 4, 10, 20, 100]:
        valid(fewer_num)

