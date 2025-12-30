# modeling_custom.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput


# === Mod 1 & 2: 模型定义 ===
class CustomFinBERT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)

        # 融合维度 (Mod 2)
        self.fusion_dim = config.hidden_size * 3

        # 增强头 (Mod 1 - 去除 BatchNorm 版)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size, config.num_labels)
        )
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True
        )
        # 层融合
        h12 = outputs.hidden_states[-1][:, 0, :]
        h11 = outputs.hidden_states[-2][:, 0, :]
        h10 = outputs.hidden_states[-3][:, 0, :]
        cls_output = torch.cat((h12, h11, h10), dim=1)

        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# === Mod 3: Trainer 定义 ===
class FocalLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # 动态获取设备
        alpha = torch.tensor([1.0, 2.0, 1.0]).to(logits.device)
        gamma = 2.0

        ce_loss = F.cross_entropy(logits, labels, reduction='none', weight=alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** gamma * ce_loss
        loss = focal_loss.mean()

        return (loss, outputs) if return_outputs else loss