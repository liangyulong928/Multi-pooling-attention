from utils import token_location_redirection


def buildModelSet(json_file, event_schema, few_num, tokenizer):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    trigger_start_index = []
    trigger_end_index = []
    answer_start_index = []
    answer_end_index = []
    i = 0
    for jsonl in json_file:
        if i % few_num == 0:
            for event in jsonl['event_list']:
                event_type, trigger, trigger_index = event['event_type'], event['trigger'], event['trigger_start_index']
                sentence_pro = jsonl['text'][:trigger_index] + '[SEP]' + \
                               jsonl['text'][trigger_index: trigger_index + len(trigger)] + '[SEP]' + \
                               jsonl['text'][trigger_index + len(trigger):]
                argument_index = {}
                trigger = '[SEP]' + trigger + '[SEP]'
                for argument in event['arguments']:
                    argument_index[argument['role']] = argument['argument']
                arguments = event_schema[event['event_type']]
                for argument in arguments:
                    question = argument
                    context = question + '[SEP]' + sentence_pro
                    token = tokenizer.encode_plus(context[:254], max_length=256, padding='max_length')
                    token_list = tokenizer.tokenize(context)
                    input_ids.append(token['input_ids'])
                    token_type_ids.append(token['token_type_ids'])
                    attention_mask.append(token['attention_mask'])
                    trigger_index_start, trigger_index_end = token_location_redirection(token_list, trigger, tokenizer)
                    print(token_list[trigger_index_start:trigger_index_end])
                    trigger_start_index.append(trigger_index_start + 1)
                    trigger_end_index.append(trigger_index_end + 1)
                    argument = argument_index.get(argument, None)
                    if argument:
                        answer_index_start, answer_index_end = token_location_redirection(token_list, argument, tokenizer)
                        print(token_list[answer_index_start:answer_index_end])
                        answer_start_index.append(answer_index_start + 1)
                        answer_end_index.append(answer_index_end + 1)
                    else:
                        answer_start_index.append(0)
                        answer_end_index.append(0)
        i += 1
    return input_ids, token_type_ids, attention_mask, trigger_start_index, trigger_end_index, answer_start_index, answer_end_index
