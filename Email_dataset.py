from transformers import BertTokenizerFast, BeitFeatureExtractor, BeitForMaskedImageModeling
from transformers import CLIPTokenizerFast, CLIPFeatureExtractor
from transformers import ViltProcessor,  ViltFeatureExtractor
import pandas as pd
import torch
import os
from PIL import Image
from torchvision.transforms import Resize, RandomResizedCrop, Normalize, Compose, CenterCrop, PILToTensor, ToTensor
from torch.utils.data import DataLoader
import random
from torchvision.datasets import ImageFolder
from utils import SplitData
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer

# from dall_e import map_pixels, load_model
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class EDPDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, data_df):
        super(EDPDataset, self).__init__()
        self.data_path = data_path
        self.data = data_df

    def __getitem__(self, item):
        text = self.data.iloc[item, 0]
        pic_path = os.path.join(self.data_path, self.data.iloc[item, 1])
        label = self.data.iloc[item, 2]
        pic = Image.open(pic_path)
        return text, pic, label

    def __len__(self):
        return len(self.data)


class EDPCollator:
    def __init__(self, tokenizer=None, feature_extractor=None):
        self.tokenizer = tokenizer if tokenizer is not None else BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
        self.feature_extractor = feature_extractor if feature_extractor is not None else BeitFeatureExtractor.from_pretrained('microsoft/dit-base')

    def text_process(self, text_list):
        text_tensor = self.tokenizer(text_list, return_tensors='pt', max_length=256, truncation=True,
                                     padding='max_length')
        return text_tensor

    def picture_process(self, picture_list):
        pixel_values = self.feature_extractor(picture_list, return_tensors='pt')
        return pixel_values

    def __call__(self, data):
        text_list, picture_list, label_list = zip(*data)
        text_tensor = self.text_process(list(text_list))
        pixel_values = self.picture_process(picture_list)
        labels = torch.LongTensor(label_list)
        inputs = dict()
        inputs.update(text_tensor)
        inputs.update(pixel_values)
        inputs['labels'] = labels
        return inputs


class EDPCollatorCLIP(EDPCollator):
    def __init__(self):
        super(EDPCollatorCLIP, self).__init__()
        self.tokenizer = CLIPTokenizerFast.from_pretrained('openai/clip-vit-base-patch32')
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

    def text_process(self, text_list):
        return self.tokenizer(text_list, return_tensors='pt', max_length=77, truncation=True, padding=True).data

    def picture_process(self, picture_list):
        return self.feature_extractor(picture_list, return_tensors='pt').data


class EDPCollatorViLT(EDPCollator):
    def __init__(self):
        super(EDPCollatorViLT, self).__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained("dandelin/vilt-b32-mlm")
        self.feature_extractor = ViltFeatureExtractor.from_pretrained("dandelin/vilt-b32-mlm")
        self.croper = CenterCrop(224)

    def text_process(self, text_list):
        return self.tokenizer(text_list, return_tensors='pt', max_length=40, truncation=True, padding=True).data

    def picture_process(self, picture_list):
        picture_list = [self.croper(_) for _ in picture_list]
        return self.feature_extractor(picture_list, return_tensors='pt').data


class EDPCollatorMMAMF(EDPCollator):
    def __init__(self):
        super(EDPCollatorMMAMF, self).__init__()
        self.tokenizer = TokenizerLSTM()
        self.feature_extractor = FeatureExtractorCNN()

    def text_process(self, text_list):
        return self.tokenizer(text_list)

    def picture_process(self, picture_list):
        return self.feature_extractor(picture_list)


class EDPTextCollator(EDPCollator):
    def __call__(self, data):
        inputs = super(EDPTextCollator, self).__call__(data)
        inputs.pop('pixel_values')
        return inputs


class EDPPictureCollator(EDPCollator):
    def __call__(self, data):
        text_list, picture_list, label_list = zip(*data)
        pixel_values = self.feature_extractor(picture_list, return_tensors='pt')
        labels = torch.LongTensor(label_list)
        inputs = dict()
        inputs.update(pixel_values)
        inputs['labels'] = labels
        return inputs


class FeatureExtractorCNN:
    def __init__(self):
        self.transform = Compose([
            CenterCrop(128),
            ToTensor()
        ])

    def __call__(self, pic_list, return_tensors='pt'):
        pic_tensors = [self.transform(_) for _ in pic_list]
        pixel_values = torch.stack(pic_tensors, dim=0)
        return {"pixel_values": pixel_values}


class TokenizerLSTM:
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

    def __call__(self, text_list, return_tensors='pt', max_length=256, truncation=True, padding='max_length'):
        vector_list = [self.sentence2vector(_) for _ in text_list]
        return {'input_ids': torch.stack(vector_list, dim=0)}


def collate(data):
    pixel_values_list = []
    bool_masked_pos_list = []
    labels_list = []
    for item in data:
        pixel_values_list.append(item['pixel_values'])
        bool_masked_pos_list.append(item['bool_masked_pos'])
        labels_list.append(item['labels'])
    return {"pixel_values": torch.stack(pixel_values_list), "bool_masked_pos": torch.stack(bool_masked_pos_list),
            "labels": torch.cat(labels_list, dim=0)}


class Extractor:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, data):
        return torch.squeeze(self.feature_extractor(data, return_tensors="pt").pixel_values)


class RawEmailDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, csv):
        self.data_path = data_path
        data = pd.read_csv(os.path.join(data_path, csv))
        self.len = len(data)
        data.fillna('null', inplace=True)
        data['pictures'] = data['file_names'].map(lambda x: x.split('.')[0] + '.jpg')
        self.texts = data['texts'].tolist()
        self.pics = data['pictures'].tolist()
        self.labels = torch.LongTensor([data['labels'].tolist()]).T
        del data

    def __getitem__(self, item):
        label = self.labels[item]
        dir = 'ham_pics' if label.item() == 0 else 'spam_pics'
        pic_path = os.path.join(self.data_path, dir, self.pics[item])
        image = Image.open(pic_path)
        # texts, pics, labels
        return (self.texts[item], image, label)

    def __len__(self):
        return self.len


class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, csv):
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
        feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/dit-base')
        self.data_path = data_path
        data = pd.read_csv(os.path.join(data_path, csv))
        self.len = len(data)
        data.fillna('null', inplace=True)
        data['pictures'] = data['file_names'].map(lambda x: x.split('.')[0] + '.jpg')
        # self.texts = self.data['texts'].tolist()
        self.pics = data['pictures'].tolist()
        self.text_tensors = tokenizer(data['texts'].tolist(), return_tensors='pt', max_length=256, truncation=True, padding='max_length')
        self.labels = torch.LongTensor([data['labels'].tolist()]).T
        self.transforms = Compose([
            CenterCrop(feature_extractor.size),
            Extractor(feature_extractor),
            # Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
        ])
    def __getitem__(self, item):
        # text = self.texts[item]
        input_ids = self.text_tensors['input_ids'][item]
        token_type_ids = self.text_tensors['token_type_ids'][item]
        attention_mask = self.text_tensors['attention_mask'][item]
        label = self.labels[item]
        dir = 'ham_pics' if label.item() == 0 else 'spam_pics'
        pic_path = os.path.join(self.data_path, dir, self.pics[item])
        image = Image.open(pic_path)
        pixel_values = self.transforms(img=image)
        return {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask, "pixel_values": pixel_values, "labels": label}

    def __len__(self):
        return self.len


class Email_text_dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        df.fillna('null', inplace=True)
        # self.texts = df['texts'].tolist()
        # self.labels = df['labels'].tolist()
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
        self.inputs = tokenizer(df['texts'].tolist(), return_tensors='pt', max_length=256, truncation=True, padding='max_length')
        self.labels = torch.LongTensor([df['labels'].tolist()]).T
        self.len = len(df)

    def __getitem__(self, item):
        # return {"text": self.texts[item], "labels": self.labels[item]}
        input_ids = self.inputs['input_ids'][item]
        token_type_ids = self.inputs['token_type_ids'][item]
        attention_mask = self.inputs['attention_mask'][item]
        label = self.labels[item]
        return {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask,
                "labels": label}

    def __len__(self):
        return self.len


# class Email_text_pretrain_dataset(torch.utils.data.Dataset):
#     # this is for mlm
#     def __init__(self, csv_path):
#         df = pd.read_csv(csv_path)
#         df.fillna('null', inplace=True)
#         texts = df['texts'].tolist()
#         tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
#         inputs = tokenizer(texts, return_tensors='pt', max_length=256, truncation=True, padding='max_length')
#         inputs['labels'] = inputs.input_ids.detach().clone()
#         # create random array of floats with equal dimensions to input_ids tensor
#         rand = torch.rand(inputs.input_ids.shape)
#         # create mask array
#         mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)
#
#         selection = []
#         for i in range(inputs.input_ids.shape[0]):
#             selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())
#         for i in range(inputs.input_ids.shape[0]):
#             inputs.input_ids[i, selection[i]] = 103
#         self.inputs = inputs
#         del df, texts, tokenizer, inputs, rand, mask_arr, selection
#
#     def __getitem__(self, item):
#         return {k: v[item] for k, v in self.inputs.items()}
#
#     def __len__(self):
#         return self.inputs['input_ids'].shape[0]



class Email_image_dataset(ImageFolder):
    def __getitem__(self, item):
        data, label = super().__getitem__(index=item)
        return {"pixel_values": data, "labels": torch.LongTensor([label])}


# class Email_image_pretrain_dataset(ImageFolder):
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     encoder = load_model("encoder.pkl", device)
#     target_image_size = 112
#     resizer = Resize(target_image_size)
#     mask_template = torch.cat((torch.zeros(98), torch.ones(98)), dim=0)
#
#
#     def preprocess(self, img):
#         img = self.resizer(img)
#         img = TF.center_crop(img, output_size=2 * [self.target_image_size])
#         img = torch.unsqueeze(T.ToTensor()(img), 0)
#         return map_pixels(img)
#
#     def __getitem__(self, index: int) -> dict[str, Any]:
#         path, target = self.samples[index]
#         sample = self.loader(path)
#
#         with torch.no_grad():
#             image = self.preprocess(sample)
#             input_ids = torch.squeeze(torch.argmax(self.encoder(image.to(self.device)), axis=1).flatten(1))
#             # bool_masked_pos = torch.randint(low=0, high=2, size=input_ids.shape).bool().to(self.device)
#             idx = torch.randperm(self.mask_template.nelement())
#             bool_masked_pos = self.mask_template[idx].bool()
#             labels = input_ids[bool_masked_pos].to(torch.device('cpu'))
#
#         if self.transform is not None:
#             pixel_values = self.transform(sample)
#
#         return {"pixel_values": pixel_values.to(torch.device("cpu")), "bool_masked_pos": bool_masked_pos, "labels": labels}


if __name__ == '__main__':
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
    # dataset = Email_text_pretrain_dataset('/home/ria/Datasets/email_data/train.csv')
    split_data = SplitData('DATA/email_data/EDP.csv', 5)
    train_df, test_df = split_data()
    train_dataset = EDPDataset('DATA/email_data/pics', train_df)
    test_dataset = EDPDataset('DATA/email_data/pics', test_df)
    # print(len(train_dataset))
    # print(len(test_dataset))
    # print(train_dataset[0])

    # collator = EDPPictureCollator(feature_extractor=FeatureExtractorCNN())
    collator = EDPTextCollator(tokenizer=TokenizerLSTM())

    dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=collator)
    for batch in dataloader:
        print(batch)
        # print(batch['pixel_values'].shape)
        print(batch['input_ids'].shape)
        break


    # for i in range(len(train_dataset)):
    #     print(train_dataset[i])
    # print('-' * 50)
    # for i in range(len(test_dataset)):
    #     print(test_dataset[i])


    # for i in range(5):
    #     print(tokenizer.decode(dataset[i]['input_ids'].tolist()))
    #     print(tokenizer.decode(dataset[i]['labels'].tolist()))

    # feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/dit-base')
    # transforms = Compose([
    #     CenterCrop(feature_extractor.size),
    #     Extractor(feature_extractor),
    #     # Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    # ])
    # dataset = Email_image_pretrain_dataset("/home/ria/Datasets/email_data/train", transform=transforms)
    # print(dataset[0]['pixel_values'].shape)
    # print(dataset[0]['bool_masked_pos'].shape)
    # print(dataset[0]['labels'].shape)
    #
    # dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate)
    # model = BeitForMaskedImageModeling.from_pretrained("microsoft/dit-base")
    # for batch in dataloader:
    #     print(batch['pixel_values'].shape)
    #     print(batch['bool_masked_pos'].shape)
    #     print(batch['labels'].shape)
    #     output = model(**batch)
    #     print(output)
    #     break


    # dataset = EmailDataset("/home/ria/桌面/final_data", 'train.csv', tokenizer, feature_extractor)
    # print(len(dataset))
    #
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    # for i in range(100):
    #     s = random.randint(0, len(dataset))
    #     batch = dataset[s]
    #     ids = batch[0].tolist()
    #     out = tokenizer.decode(token_ids=ids)
    #     print(out)



    # for batch in dataloader:
    #     a, b, c, d, e = batch
    #     print(a.shape)
    #     print(b.shape)
    #     print(c.shape)
    #     print(d.shape)
    #     print(e.shape)
    #     print(e.dtype)
    #     break

