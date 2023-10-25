from models import MMTD, ViltEmailModel, CLIPEmailModel, MMA_MF
import torch
from Email_dataset import EDPDataset, EDPCollator, FeatureExtractorCNN, TokenizerLSTM
from utils import metrics, SplitData, EvalMetrics
from transformers import Trainer, TrainingArguments
from transformers import BertTokenizerFast, BeitFeatureExtractor
import wandb
import os
import random
import numpy as np
from PIL import Image
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
from torchvision.transforms import CenterCrop, ToTensor, Compose
from transformers import CLIPTokenizerFast, CLIPFeatureExtractor, ViltFeatureExtractor
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def add_gaussian_noise(image, mean=0, std=25):
    """
    Add Gaussian noise to an image.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        mean (int): Mean value of the Gaussian noise.
        std (int): Standard deviation of the Gaussian noise.

    Returns:
        numpy.ndarray: Image with added Gaussian noise.
    """
    img_array = np.array(image)
    noise = np.random.normal(mean, std, img_array.shape).astype(np.uint8)
    noisy_image_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    noisy_image = Image.fromarray(noisy_image_array)
    return noisy_image

# Example usage:
# noisy_image = add_gaussian_noise(input_image, mean=0, std=25)



def add_text_noise(text, noise_level=0.1):
    """
    Add random noise to the input text.

    Args:
        text (str): The input text to which noise will be added.
        noise_level (float): The proportion of characters to add noise to (between 0 and 1).

    Returns:
        str: The text with added noise.
    """
    if noise_level < 0 or noise_level > 1:
        raise ValueError("noise_level should be between 0 and 1")

    noisy_text = list(text)
    num_noise_chars = int(len(text) * noise_level)

    for _ in range(num_noise_chars):
        index = random.randint(0, len(text) - 1)
        noisy_text[index] = random.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_-+=[]{}|;:,.<>?")

    return ''.join(noisy_text)

# Example usage:
# input_text = "This is a sample text."
# noisy_text = add_text_noise(input_text, noise_level=0.2)
# print(noisy_text)


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
        # text_list = [add_text_noise(_) for _ in text_list]
        # picture_list = [add_gaussian_noise(_) for _ in picture_list]
        text_tensor = self.text_process(list(text_list))
        pixel_values = self.picture_process(picture_list)
        labels = torch.LongTensor(label_list)
        inputs = dict()
        inputs.update(text_tensor)
        inputs.update(pixel_values)
        inputs['labels'] = labels
        return inputs


class EDPCollatorMMAMF(EDPCollator):
    def __init__(self):
        super(EDPCollatorMMAMF, self).__init__()
        self.tokenizer = TokenizerLSTM()
        self.feature_extractor = FeatureExtractorCNN()

    def text_process(self, text_list):
        text_list = [add_text_noise(_) for _ in text_list]
        return self.tokenizer(text_list)

    def picture_process(self, picture_list):
        picture_list = [add_gaussian_noise(_) for _ in picture_list]
        return self.feature_extractor(picture_list)


class EDPCollatorViLT(EDPCollator):
    def __init__(self):
        super(EDPCollatorViLT, self).__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained("dandelin/vilt-b32-mlm")
        self.feature_extractor = ViltFeatureExtractor.from_pretrained("dandelin/vilt-b32-mlm")
        self.croper = CenterCrop(224)

    def text_process(self, text_list):
        text_list = [add_text_noise(_) for _ in text_list]
        return self.tokenizer(text_list, return_tensors='pt', max_length=40, truncation=True, padding=True).data

    def picture_process(self, picture_list):
        picture_list = [add_gaussian_noise(_) for _ in picture_list]
        picture_list = [self.croper(_) for _ in picture_list]
        return self.feature_extractor(picture_list, return_tensors='pt').data
        return out


class EDPCollatorCLIP(EDPCollator):
    def __init__(self):
        super(EDPCollatorCLIP, self).__init__()
        self.tokenizer = CLIPTokenizerFast.from_pretrained('openai/clip-vit-base-patch32')
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

    def text_process(self, text_list):
        text_list = [add_text_noise(_) for _ in text_list]
        return self.tokenizer(text_list, return_tensors='pt', max_length=77, truncation=True, padding=True).data

    def picture_process(self, picture_list):
        picture_list = [add_gaussian_noise(_) for _ in picture_list]
        return self.feature_extractor(picture_list, return_tensors='pt').data

#-----------------------------------------------------------------------------------------------------------------------


fold = 5
split_data = SplitData('DATA/email_data/EDP.csv', fold)

for i in range(fold):
    train_df, test_df = split_data()
    train_dataset = EDPDataset('DATA/email_data/pics', train_df)
    test_dataset = EDPDataset('DATA/email_data/pics', test_df)

    # model = MMTD(bert_pretrain_weight='output/bert/checkpoints/fold1/checkpoint-4683',
    #              beit_pretrain_weight='output/dit/checkpoints/fold1/checkpoint-1562')
    # model.load_state_dict(torch.load(f'output/MMTD/checkpoints/fold{i+1}/checkpoint-939/pytorch_model.bin'))
    # model = MMA_MF()
    # model.load_state_dict(torch.load(f'output/mma_mf/checkpoints/fold{i+1}/checkpoint-5865/pytorch_model.bin'))
    # model = CLIPEmailModel()
    # model.load_state_dict(torch.load(f'output/clip/checkpoints/fold{i+1}/checkpoint-15610/pytorch_model.bin'))
    model = ViltEmailModel()
    model.load_state_dict(torch.load(f'output/vilt/checkpoints/fold{i+1}/checkpoint-15610/pytorch_model.bin'))

    model.eval()

    name = f'robustness_vilt{i+1}'
    args = TrainingArguments(
        output_dir=f'./output/{name}/',
        logging_dir=f'./output/{name}/log',
        logging_strategy='epoch',
        learning_rate=5e-4,
        per_device_train_batch_size=40,
        per_device_eval_batch_size=40,
        num_train_epochs=3,
        weight_decay=0.0,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        run_name=name,
        auto_find_batch_size=False,
        overwrite_output_dir=True,
        save_total_limit=5,
        remove_unused_columns=False,
        # report_to=["wandb"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=EDPCollatorViLT(),
        compute_metrics=metrics,
    )

    trainer.compute_metrics = EvalMetrics(f'./output/{name}', name, True)
    test_acc = trainer.evaluate(eval_dataset=test_dataset)
    test_result = {'test_acc': test_acc['eval_acc'], 'test_loss': test_acc['eval_loss']}
    print(test_result)