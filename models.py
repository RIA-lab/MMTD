from transformers import BertForSequenceClassification, BeitForImageClassification, BeitConfig, BertConfig
from transformers.models.bert.modeling_bert import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss
import torch
from transformers import CLIPModel, CLIPConfig
from transformers import ViltModel, ViltConfig


class MMTD(torch.nn.Module):
    def __init__(self, bert_cfg=BertConfig(), beit_cfg=BeitConfig(), bert_pretrain_weight=None, beit_pretrain_weight=None):
        super(MMTD, self).__init__()
        self.text_encoder = BertForSequenceClassification.from_pretrained(bert_pretrain_weight) if bert_pretrain_weight is not None else BertForSequenceClassification(bert_cfg)
        self.image_encoder = BeitForImageClassification.from_pretrained(beit_pretrain_weight) if beit_pretrain_weight is not None else BeitForImageClassification(beit_cfg)
        self.text_encoder.config.output_hidden_states = True
        self.image_encoder.config.output_hidden_states = True
        self.multi_modality_transformer_layer = torch.nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.pooler = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.Tanh()
        )
        # self.dropout = torch.nn.Dropout(p=0.1)
        self.classifier = torch.nn.Linear(768, 2)
        self.num_labels = 2
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, input_ids, token_type_ids, attention_mask, pixel_values, labels=None):
        text_outputs = self.text_encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        text_last_hidden_state = text_outputs.hidden_states[12]
        image_last_hidden_state = image_outputs.hidden_states[12]
        text_last_hidden_state += torch.zeros(text_last_hidden_state.size()).to(self.device)
        image_last_hidden_state += torch.ones(image_last_hidden_state.size()).to(self.device)
        fuse_hidden_state = torch.cat([text_last_hidden_state, image_last_hidden_state], dim=1)
        outputs = self.multi_modality_transformer_layer(fuse_hidden_state)
        outputs = self.pooler(outputs[:, 0, :])
        logits = self.classifier(outputs)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )



class BertBeitEmailModelNoCLS(torch.nn.Module):
    def __init__(self, bert_cfg=BertConfig(), beit_cfg=BeitConfig(), bert_pretrain_weight=None, beit_pretrain_weight=None):
        super(BertBeitEmailModelNoCLS, self).__init__()
        self.text_encoder = BertForSequenceClassification.from_pretrained(bert_pretrain_weight) if bert_pretrain_weight is not None else BertForSequenceClassification(bert_cfg)
        self.image_encoder = BeitForImageClassification.from_pretrained(beit_pretrain_weight) if beit_pretrain_weight is not None else BeitForImageClassification(beit_cfg)
        self.text_encoder.config.output_hidden_states = True
        self.image_encoder.config.output_hidden_states = True
        self.multi_modality_transformer_layer = torch.nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.squeeze_layer = torch.nn.Linear(768, 1)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.classifier = torch.nn.Linear(453, 2)
        self.num_labels = 2
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, input_ids, token_type_ids, attention_mask, pixel_values, labels=None):
        text_outputs = self.text_encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        text_last_hidden_state = text_outputs.hidden_states[12]
        image_last_hidden_state = image_outputs.hidden_states[12]
        text_last_hidden_state += torch.zeros(text_last_hidden_state.size()).to(self.device)
        image_last_hidden_state += torch.ones(image_last_hidden_state.size()).to(self.device)
        fuse_hidden_state = torch.cat([text_last_hidden_state, image_last_hidden_state], dim=1)
        outputs = self.multi_modality_transformer_layer(fuse_hidden_state)
        outputs = torch.squeeze(self.squeeze_layer(outputs))
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )



class BertBeitEmailModelFc(torch.nn.Module):
    def __init__(self, bert_cfg=BertConfig(), beit_cfg=BeitConfig(), bert_pretrain_weight=None, beit_pretrain_weight=None):
        super(BertBeitEmailModelFc, self).__init__()
        self.text_encoder = BertForSequenceClassification.from_pretrained(bert_pretrain_weight) if bert_pretrain_weight is not None else BertForSequenceClassification(bert_cfg)
        self.image_encoder = BeitForImageClassification.from_pretrained(beit_pretrain_weight) if beit_pretrain_weight is not None else BeitForImageClassification(beit_cfg)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.Linear(64, 2)
        )
        self.num_labels = 2
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, input_ids, token_type_ids, attention_mask, pixel_values, labels=None):
        text_outputs = self.text_encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        hidden_state = torch.cat([text_outputs.logits, image_outputs.logits], dim=1)
        logits = self.classifier(hidden_state)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )



class CLIPEmailModel(CLIPModel):
    def __init__(self, config=CLIPConfig()):
        super(CLIPEmailModel, self).__init__(config=config)
        self.multi_modality_transformer_layer = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.pooler = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.Tanh()
        )
        self.classifier = torch.nn.Linear(512, 2)
        self.num_labels = 2


    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        out = super(CLIPEmailModel, self).forward(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        text_last_hidden_state = out.text_model_output.last_hidden_state
        text_last_hidden_state = self.text_projection(text_last_hidden_state)
        image_last_hidden_state = out.vision_model_output.last_hidden_state
        image_last_hidden_state512 = self.visual_projection(image_last_hidden_state)
        image_last_hidden_state512 += torch.ones(image_last_hidden_state512.size()).to(self.device)
        fuse_hidden_state = torch.cat([text_last_hidden_state, image_last_hidden_state512], dim=1)
        outputs = self.multi_modality_transformer_layer(fuse_hidden_state)
        outputs = self.pooler(outputs[:, 0, :])
        logits = self.classifier(outputs)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

class ViltEmailModel(ViltModel):
    def __init__(self, config=ViltConfig()):
        super(ViltEmailModel, self).__init__(config=config)
        self.classifier = torch.nn.Linear(768, 2)
        self.num_labels = 2
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, pixel_values=None, pixel_mask=None, labels=None):
        out = super(ViltEmailModel, self).forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, pixel_values=pixel_values, pixel_mask=pixel_mask)
        logits = self.classifier(out.pooler_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), padding="same"),
            torch.nn.BatchNorm2d(num_features=32, eps=1e-6, momentum=0.9),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding="same"),
            torch.nn.BatchNorm2d(num_features=64, eps=1e-6, momentum=0.9),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding="same"),
            torch.nn.BatchNorm2d(num_features=64, eps=1e-6, momentum=0.9),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Linear(16384, 64),
            torch.nn.BatchNorm1d(num_features=64, eps=1e-6),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
            torch.nn.Softmax()
        )
    def forward(self, pixel_values, labels=None):
        out = self.layer1(pixel_values)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.flatten(out, start_dim=1)
        logits = self.layer4(out)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

class LSTM(torch.nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm1 = torch.nn.LSTM(input_size=200, hidden_size=64, batch_first=True, dropout=0.3)
        self.lstm2 = torch.nn.LSTM(input_size=64, hidden_size=32, batch_first=True, dropout=0.3)
        self.fc = torch.nn.Linear(in_features=32, out_features=2, bias=True)


    def forward(self, input_ids, labels=None):
        out1, _ = self.lstm1(input_ids)
        out2, _ = self.lstm2(out1)
        out = self.fc(out2[:, -1, :])
        logits = torch.nn.functional.softmax(out, dim=1)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

class MMA_MF(torch.nn.Module):
    def __init__(self):
        super(MMA_MF, self).__init__()
        self.ltsm = LSTM()
        self.cnn = CNN()
        self.fc = torch.nn.Linear(4, 64)
        self.classifier = torch.nn.Linear(64, 2)

    def forward(self, input_ids, pixel_values, labels=None):
        lstm_out = self.ltsm(input_ids)
        cnn_out = self.cnn(pixel_values)
        lstm_out = lstm_out.logits
        cnn_out = torch.nn.functional.softmax(cnn_out.logits, dim=1)
        out = torch.cat([lstm_out, cnn_out], dim=1)
        out = self.fc(out)
        out = torch.nn.functional.relu(out)
        logits = self.classifier(out)
        logits = torch.nn.functional.sigmoid(logits)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
