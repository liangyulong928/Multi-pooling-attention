from torch.utils.data import Dataset


class MPADataset(Dataset):
    def __init__(self, input_ids, token_type_ids, attention_mask, trigger_start, trigger_end, answer_start, answer_end):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.trigger_start = trigger_start
        self.trigger_end = trigger_end
        self.answer_start = answer_start
        self.answer_end = answer_end

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        attention_mask = self.attention_mask[index]
        token_type_ids = self.token_type_ids[index]
        trigger_start = self.trigger_start[index]
        trigger_end = self.trigger_end[index]
        answer_start = self.answer_start[index]
        answer_end = self.answer_end[index]

        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'trigger_start': trigger_start,
            'trigger_end': trigger_end,
            'answer_start': answer_start,
            'answer_end': answer_end
        }

    def __len__(self):
        return len(self.input_ids)
