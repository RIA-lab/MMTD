from Email_dataset import EDPDataset, EDPCollator
from models import MMTD
from transformers import Trainer, TrainingArguments
from torch.optim import AdamW, lr_scheduler
from utils import metrics, SplitData, save_config, EvalMetrics
import wandb
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
fold = 5
split_data = SplitData('DATA/email_data/EDP.csv', fold)

bert_checkpoint_path = './output/bert/checkpoints'
bert_folds = os.listdir(bert_checkpoint_path)
bert_checkpoints = list()
for _ in os.listdir(bert_checkpoint_path):
    bert_checkpoints.extend(os.listdir(os.path.join(bert_checkpoint_path, _)))

dit_checkpoint_path = './output/dit/checkpoints'
dit_folds = os.listdir(dit_checkpoint_path)
dit_checkpoints = list()
for _ in os.listdir(dit_checkpoint_path):
    dit_checkpoints.extend(os.listdir(os.path.join(dit_checkpoint_path, _)))

if __name__ == '__main__':
    for i in range(fold):
        wandb.init(project='MMTD')
        wandb.run.name = 'MMTD-fold-' + str(i + 1)
        train_df, test_df = split_data()
        train_dataset = EDPDataset('DATA/email_data/pics', train_df)
        test_dataset = EDPDataset('DATA/email_data/pics', test_df)

        bert_checkpoint = os.path.join(bert_checkpoint_path, bert_folds[i], bert_checkpoints[i])
        dit_checkpoint = os.path.join(dit_checkpoint_path, dit_folds[i], dit_checkpoints[i])
        model = MMTD(bert_pretrain_weight=bert_checkpoint, beit_pretrain_weight=dit_checkpoint)

        for p in model.text_encoder.parameters():
            p.requires_grad = False
        for p in model.image_encoder.parameters():
            p.requires_grad = False

        filtered_parameters = []
        for p in filter(lambda p: p.requires_grad, model.parameters()):
            filtered_parameters.append(p)

        optimizer = AdamW(filtered_parameters, lr=5e-4)

        args = TrainingArguments(
            output_dir='./output/MMTD/checkpoints/fold' + str(i + 1),
            logging_dir='./output/MMTD/log',
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
            run_name=wandb.run.name,
            auto_find_batch_size=False,
            overwrite_output_dir=True,
            save_total_limit=5,
            remove_unused_columns=False,
            report_to=["wandb"],
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            optimizers=(optimizer, None),
            data_collator=EDPCollator(),
            compute_metrics=metrics,
        )

        trainer.train()

        trainer.train()
        train_acc = trainer.evaluate(eval_dataset=train_dataset)
        train_result = {'train_acc': train_acc['eval_acc'], 'train_loss': train_acc['eval_loss']}
        wandb.log(train_result)

        trainer.compute_metrics = EvalMetrics('./output/MMTD/results', wandb.run.name, True)
        test_acc = trainer.evaluate(eval_dataset=test_dataset)
        test_result = {'test_acc': test_acc['eval_acc'], 'test_loss': test_acc['eval_loss']}
        wandb.log(test_result)

        wandb.config = args.to_dict()
        save_config(args.to_dict(), os.path.join('./output/MMTD/configs', wandb.run.name + '.yaml'))
        wandb.finish()