import json
import logging
import os
import pickle

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from MultiPoolingAttentionQA import BertMultiPoolingAttentionForQuestionAnswering
from MPADataUtils import MPADataset
from DataProcessing import buildModelSet
from Utils import convert_file_to_json, get_event_schema

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizer.from_pretrained("chinese-bert-wwm-ext")

n_epoch = 10
batch_size = 12
learning_rate = 5e-5
adam_epsilon = 1e-8
warmup_steps = 0
max_grad_norm = 1.0
source_data_path = './data/origin/DuEE1.0/'
encode_data_path = './data/process/DuEE1.0/'
save_model_path = './output/model/MPA/'


def train(few_num):
    model = BertMultiPoolingAttentionForQuestionAnswering.from_pretrained("chinese-bert-wwm-ext")
    event_schema = get_event_schema(source_data_path, encode_data_path)


    if os.path.isfile(encode_data_path + f'duee_train_MultiPoolingAttention_few_{few_num}.pkl'):
        with open(encode_data_path + f'duee_train_MultiPoolingAttention_few_{few_num}.pkl', 'rb') as f:
            dataset = pickle.load(f)
    else:
        if not os.path.isfile(source_data_path + f"duee_train.json/duee_train.json"):
            raise FileNotFoundError("The target data cannot be generated because the source data does not exist")
        else:
            input_ids, token_type_ids, attention_mask, trigger_start_index, trigger_end_index, answer_start_index, answer_end_index = buildModelSet(
                convert_file_to_json(source_data_path + f"duee_train.json/duee_train.json"),
                event_schema, few_num, tokenizer)
            dataset = MPADataset(torch.tensor(input_ids, dtype=torch.long),
                                 torch.tensor(token_type_ids, dtype=torch.long),
                                 torch.tensor(attention_mask, dtype=torch.long),
                                 torch.tensor(trigger_start_index, dtype=torch.long).unsqueeze(1),
                                 torch.tensor(trigger_end_index, dtype=torch.long).unsqueeze(1),
                                 torch.tensor(answer_start_index, dtype=torch.long).unsqueeze(1),
                                 torch.tensor(answer_end_index, dtype=torch.long).unsqueeze(1))
            with open(encode_data_path + f'duee_train_MultiPoolingAttention_few_{few_num}.pkl', 'wb') as f:
                pickle.dump(dataset, f)

    if os.path.isfile(save_model_path + f"model_MultiPoolingAttention_few_{few_num}.pth"):
        model.load_state_dict(torch.load(save_model_path + f"model_MultiPoolingAttention_few_{few_num}.pth",
                                         map_location=torch.device('cpu')), False)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model.to(device)
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        logging.basicConfig(filename=save_model_path + f'train_MultiPoolingAttention_few_{few_num}.log',
                            level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        t_total = int(n_epoch * len(train_loader))
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=learning_rate * t_total,
                                                    num_training_steps=t_total)
        print('------------------------model train---------------------------')
        model.train()
        for epoch in range(n_epoch):
            loop = tqdm(enumerate(train_loader), total=len(train_loader))
            loop.set_description(f'Epoch [{epoch + 1}/{n_epoch}]')
            for step, batch in loop:
                inputs = {'input_ids': batch['input_ids'].to(device),
                          'token_type_ids': batch['token_type_ids'].to(device),
                          'attention_mask': batch['attention_mask'].to(device),
                          'start_positions': batch['answer_start'].to(device),
                          'end_positions': batch['answer_end'].to(device),
                          'trigger_start_positions': batch['trigger_start'].to(device),
                          'trigger_end_positions': batch['trigger_end'].to(device)}
                output = model(**inputs)
                loss = output[0]
                loss = loss.mean()
                loop.set_postfix(loss=loss.item())
                if step % 10 == 0:
                    logging.info(f'epoch {epoch + 1} _ step {step + 1} : loss = {loss.item()}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
        torch.save(model.state_dict(), save_model_path + f"model_MultiPoolingAttention_few_{few_num}.pth")
        print(f"The {100 / fewer_num} % training data model was saved to '{save_model_path}model_MultiPoolingAttention_few_{few_num}.pth'")
    print("Model training completed")


if __name__ == '__main__':
    for fewer_num in [1, 2, 4, 10, 20, 100]:
        train(fewer_num)
