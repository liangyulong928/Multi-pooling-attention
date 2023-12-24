from typing import Optional, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss
from torch import nn
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers import BertModel, BertPreTrainedModel
import torch.nn.functional as F


class BertMultiPoolingAttentionForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.pooling = nn.Linear(768 * 2, 768)
        self.attention = nn.Linear(768 * 2, 256)
        self.qa_outputs = nn.Linear(config.hidden_size * 2, config.num_labels)
        self.relu = nn.ReLU()
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            start_positions: Optional[torch.Tensor] = None,
            end_positions: Optional[torch.Tensor] = None,
            trigger_start_positions: Optional[torch.Tensor] = None,
            trigger_end_positions: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_context = outputs[0]
        pooling_1 = F.max_pool1d(hidden_context.permute(0, 2, 1), kernel_size=256, stride=256).squeeze(-1)
        pooling_hidden_list = []
        trigger_start_positions = trigger_start_positions.int().tolist()
        trigger_end_positions = trigger_end_positions.int().tolist()
        for i in range(hidden_context.size(0)):
            if trigger_start_positions[i][0] < 200 and trigger_start_positions[i][0] < 200:
                s_position = trigger_start_positions[i][0]
                e_position = trigger_end_positions[i][0]
            else:
                s_position, e_position = 10, 10
            pooling_2 = F.max_pool1d(hidden_context[i][:s_position].permute(1, 0),
                                     kernel_size=s_position,
                                     stride=s_position).squeeze(-1)
            pooling_4 = F.max_pool1d(hidden_context[i][e_position:].permute(1, 0),
                                     kernel_size=256 - e_position,
                                     stride=256 - e_position).squeeze(-1)
            pooling_hidden = torch.cat((pooling_2, pooling_4), dim=0)
            pooling_hidden_list.append(pooling_hidden)
        pooling_hidden_tensor = torch.stack(pooling_hidden_list, dim=1).permute(1, 0)

        split_pooling = self.relu(self.pooling(pooling_hidden_tensor))
        pooling_attention = self.attention(torch.cat((pooling_1, split_pooling), dim=1))
        attention = F.normalize(pooling_attention, p=1, dim=1)
        attention_hidden = attention.unsqueeze(1).permute(0, 2, 1) * hidden_context
        sequence_output = torch.cat((hidden_context, attention_hidden), dim=2)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
