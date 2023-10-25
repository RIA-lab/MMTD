from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, auc, roc_curve
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
from transformers import CLIPTokenizerFast, CLIPFeatureExtractor
from transformers import ViltProcessor, BertTokenizerFast, ViltFeatureExtractor
from torchvision.transforms import Resize, CenterCrop, ToTensor, Compose
import seaborn as sb
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
import pandas as pd
import yaml
import os


class SplitData:
    def __init__(self, csv, k_fold):
        data = pd.read_csv(csv)
        data.fillna('null', inplace=True)
        quantity = int((len(data) - len(data) % 5) / 5)
        self.fold_list = [data.iloc[quantity * i:quantity * (i + 1), :] for i in range(k_fold - 1)]
        self.fold_list.append(data.iloc[(k_fold - 1) * quantity:, :])

    def __call__(self):
        test_data = self.fold_list.pop()
        train_data = pd.concat(self.fold_list, ignore_index=True)
        self.fold_list.insert(0, test_data)
        return train_data, test_data


class MMA_MFCollator():
    def __init__(self):
        self.max_length = 500
        self.tokenizer = get_tokenizer('basic_english')
        self.glove = GloVe(name='6B', dim=200)
        self.transforms = Compose([
            CenterCrop(128),
            ToTensor()
        ])

    def sentence2vector(self, sentence):
        tokens = self.tokenizer(sentence)
        if len(tokens) < self.max_length:
            tokens.extend(['<PAD>' for i in range(self.max_length - len(tokens))])
        else:
            tokens = tokens[:self.max_length]
        return self.glove.get_vecs_by_tokens(tokens=tokens)

    def __call__(self, data):
        vector_list = []
        pic_list = []
        label_list = []
        for item in data:
            vector_list.append(self.sentence2vector(item[0]))
            pic_list.append(self.transforms(item[1]))
            label_list.append(item[2])
        out = {'input_ids': torch.stack(vector_list, dim=0), 'pixel_values': torch.stack(pic_list, dim=0), 'labels': torch.stack(label_list, dim=0)}
        return out


class LSTMCollator():
    def __init__(self):
        self.max_length = 500
        self.tokenizer = get_tokenizer('basic_english')
        self.glove = GloVe(name='6B', dim=200)

    def sentence2vector(self, sentence):
        tokens = self.tokenizer(sentence)
        if len(tokens) < self.max_length:
            tokens.extend(['<PAD>' for i in range(self.max_length - len(tokens))])
        else:
            tokens = tokens[:self.max_length]
        return self.glove.get_vecs_by_tokens(tokens=tokens)

    def __call__(self, data):
        vector_list = []
        label_list = []
        for item in data:
            vector_list.append(self.sentence2vector(item[0]))
            label_list.append(item[2])
        out = {'input_ids': torch.stack(vector_list, dim=0), 'labels': torch.stack(label_list, dim=0)}
        return out

class CLIPCollator:
    def __init__(self):
        self.tokenizer = CLIPTokenizerFast.from_pretrained('openai/clip-vit-base-patch32')
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

    def __call__(self, data):
        text_list = []
        pic_list = []
        label_list = []
        for item in data:
            text_list.append(item[0])
            pic_list.append(item[1])
            label_list.append(item[2])
        out = self.tokenizer(text_list, return_tensors='pt', max_length=77, truncation=True, padding=True).data
        img_out = self.feature_extractor(pic_list, return_tensors='pt').data
        out.update(img_out)
        out['labels'] = torch.stack(label_list, dim=0)
        return out


class ViLTCollator:
    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained("dandelin/vilt-b32-mlm")
        self.feature_extractor = ViltFeatureExtractor.from_pretrained("dandelin/vilt-b32-mlm")
        # self.feature_extractor.do_resize = False
        # self.resizer = Resize(224)
        self.croper = CenterCrop(224)

    def __call__(self, data):
        text_list = []
        pic_list = []
        label_list = []
        for item in data:
            text_list.append(item[0])
            pic_list.append(self.croper(item[1]))
            label_list.append(item[2])
        out = self.tokenizer(text_list, return_tensors='pt', max_length=40, truncation=True, padding=True).data
        img_out = self.feature_extractor(pic_list, return_tensors='pt').data
        out.update(img_out)
        out['labels'] = torch.stack(label_list, dim=0)
        return out


def save_config(config: dict, save_path: str):
    with open(save_path, 'w') as f:
        f.write(yaml.dump(config, allow_unicode=True))


def metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"acc": acc}


class EvalMetrics:
    def __init__(self, save_path=None, save_name=None, heatmap=False):
        self.save_path = save_path
        self.save_name = save_name
        self.heatmap = heatmap

    def __call__(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        scores = torch.softmax(torch.from_numpy(logits).float(), dim=1)
        scores = scores.detach().numpy()
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if self.save_name is not None:
            plt.title('ROC graph of ' + self.save_name)
        else:
            plt.title('ROC graph')

        classes = ['ham', 'spam']
        colors = ['navy', 'aqua']
        for i in range(len(classes)):
            fpr, tpr, thresholds = roc_curve(labels, scores[:, i], pos_label=i)
            auc_values = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i],
                     label='ROC curve of {0} (area = {1:0.8f})'''.format(classes[i], auc_values))

        plt.legend(loc="lower right")
        if self.save_name is not None:
            plt.savefig(os.path.join(self.save_path, self.save_name + '_roc.jpg'))
        plt.show()

        acc = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions, target_names=classes)
        matrix = confusion_matrix(labels, predictions)
        if self.heatmap:
            sb.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')

            if self.save_name is not None:
                plt.title(self.save_name + '_heatmap')
                plt.savefig(os.path.join(self.save_path, self.save_name + '_heatmap.jpg'))
            else:
                plt.title('heatmap')
            plt.show()
        if self.save_name is not None:
            with open(os.path.join(self.save_path, self.save_name + '.txt'), 'w') as f:
                f.write(self.save_name + '\n')
                f.write("acc: " + str(acc))
                f.write("\n\nreport:\n")
                f.write(report)
                f.write("\nconfusion matrix:\n")
                f.write(str(matrix))
        return {"acc": acc}


def eval(dataloader, model, save_path=None, save_name=None, heatmap=False):
    loop = tqdm(dataloader)
    labels_all = []
    scores_all = []
    prediction_all = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device=device)
    for batch in loop:
        inputs = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            output = model(**inputs)

        scores = torch.softmax(output.logits, dim=1)
        scores_all.extend(scores.tolist())
        prediction = output.logits.argmax(axis=1)
        prediction_all.extend(prediction.tolist())
        labels_all.extend(inputs['labels'].flatten().tolist())
        loop.set_description('evaluate')

    scores_all = np.array(scores_all)
    labels_all = np.array(labels_all)
    prediction_all = np.array(prediction_all)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if save_name is not None:
        plt.title('ROC graph of ' + save_name)
    else:
        plt.title('ROC graph')

    classes = ['ham', 'spam']
    colors = ['navy', 'aqua']
    for i in range(len(classes)):
        fpr, tpr, thresholds = roc_curve(labels_all, scores_all[:, i], pos_label=i)
        auc_values = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], label='ROC curve of {0} (area = {1:0.8f})'''.format(classes[i], auc_values))

    plt.legend(loc="lower right")
    if save_name is not None:
        plt.savefig(os.path.join(save_path, save_name + '_roc.jpg'))
    plt.show()

    acc = accuracy_score(labels_all, prediction_all)
    report = classification_report(labels_all, prediction_all, target_names=classes)
    matrix = confusion_matrix(labels_all, prediction_all)
    if heatmap:
        sb.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        if save_name is not None:
            plt.title(save_name + '_heatmap')
            plt.savefig(os.path.join(save_path, save_name + '_heatmap.jpg'))
        else:
            plt.title('heatmap')
        plt.show()
    if save_name is not None:
        with open(os.path.join(save_path, save_name + '.txt'), 'w') as f:
            f.write(save_name + '\n')
            f.write("acc: " + str(acc))
            f.write("\nreport:\n")
            f.write(report)
            f.write("\nconfusion matrix:\n")
            f.write(str(matrix))

    # print("acc:", acc)
    # print("report:")
    # print(report)
    # print("confusion matrix:")
    # print(matrix)
