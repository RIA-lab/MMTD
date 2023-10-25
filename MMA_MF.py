from models import MMA_MF
from utils import metrics, save_config, SplitData, EvalMetrics
from transformers import Trainer, TrainingArguments
from Email_dataset import EDPDataset, EDPCollatorMMAMF
from torch.optim import AdamW
import torch
import wandb
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
fold = 5
split_data = SplitData('DATA/email_data/EDP.csv', fold)

cnn_checkpoint_path = './output/cnn/checkpoints'
cnn_folds = os.listdir(cnn_checkpoint_path)
cnn_checkpoints = list()
for _ in os.listdir(cnn_checkpoint_path):
    cnn_checkpoints.extend(os.listdir(os.path.join(cnn_checkpoint_path, _)))

lstm_checkpoint_path = './output/lstm/checkpoints'
lstm_folds = os.listdir(lstm_checkpoint_path)
lstm_checkpoints = list()
for _ in os.listdir(lstm_checkpoint_path):
    lstm_checkpoints.extend(os.listdir(os.path.join(lstm_checkpoint_path, _)))


if __name__ == '__main__':
    for i in range(fold):
        wandb.init(project='MMTD')
        wandb.run.name = 'MMA-MF-fold-' + str(i + 1)
        train_df, test_df = split_data()
        train_dataset = EDPDataset('DATA/email_data/pics', train_df)
        test_dataset = EDPDataset('DATA/email_data/pics', test_df)

        model = MMA_MF()
        cnn_checkpoint = os.path.join(cnn_checkpoint_path, cnn_folds[i], cnn_checkpoints[i], 'pytorch_model.bin')
        lstm_checkpoint = os.path.join(lstm_checkpoint_path, lstm_folds[i], lstm_checkpoints[i], 'pytorch_model.bin')
        model.ltsm.load_state_dict(torch.load(lstm_checkpoint))
        model.cnn.load_state_dict(torch.load(cnn_checkpoint))

        for p in model.ltsm.parameters():
            p.requires_grad = False
        for p in model.cnn.parameters():
            p.requires_grad = False

        filtered_parameters = []
        for p in filter(lambda p: p.requires_grad, model.parameters()):
            filtered_parameters.append(p)

        optimizer = AdamW(filtered_parameters, lr=0.001)

        args = TrainingArguments(
            output_dir='./output/mma_mf/checkpoints/fold' + str(i + 1),
            logging_dir='./output/mma_mf/log',
            logging_strategy='epoch',
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            num_train_epochs=15,
            # fp16=True,
            learning_rate=1e-3,
            weight_decay=0.0,
            remove_unused_columns=False,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            run_name=wandb.run.name,
            auto_find_batch_size=False,
            overwrite_output_dir=True,
            save_total_limit=5,
            report_to=["wandb"],
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=EDPCollatorMMAMF(),
            compute_metrics=metrics,
            optimizers=(optimizer, None)
        )

        trainer.train()
        train_acc = trainer.evaluate(eval_dataset=train_dataset)
        train_result = {'train_acc': train_acc['eval_acc'], 'train_loss': train_acc['eval_loss']}
        wandb.log(train_result)

        trainer.compute_metrics = EvalMetrics('output/mma_mf/results', wandb.run.name, True)
        test_acc = trainer.evaluate(eval_dataset=test_dataset)
        test_result = {'test_acc': test_acc['eval_acc'], 'test_loss': test_acc['eval_loss']}
        wandb.log(test_result)

        wandb.config = args.to_dict()
        save_config(args.to_dict(), os.path.join('./output/mma_mf/configs', wandb.run.name + '.yaml'))
        wandb.finish()
        del model, args, trainer

