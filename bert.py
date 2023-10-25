import wandb
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from Email_dataset import EDPDataset, EDPTextCollator
from utils import metrics, SplitData, save_config, EvalMetrics
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
fold = 5
split_data = SplitData('DATA/email_data/EDP.csv', fold)

if __name__ == '__main__':
    for i in range(fold):
        wandb.init(project='MMTD')
        wandb.run.name = 'bert-fold-' + str(i + 1)
        train_df, test_df = split_data()
        train_dataset = EDPDataset('DATA/email_data/pics', train_df)
        test_dataset = EDPDataset('DATA/email_data/pics', test_df)
        model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)

        args = TrainingArguments(
            output_dir='./output/bert/checkpoints/fold' + str(i + 1),
            logging_dir='./output/bert/log',
            logging_strategy='epoch',
            learning_rate=5e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=3,
            # weight_decay=0.0,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            run_name=wandb.run.name,
            auto_find_batch_size=False,
            overwrite_output_dir=True,
            save_total_limit=8,
            remove_unused_columns=False,
            report_to=["wandb"],
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=EDPTextCollator(),
            compute_metrics=metrics,
        )

        trainer.train()

        train_acc = trainer.evaluate(eval_dataset=train_dataset)
        train_result = {'train_acc': train_acc['eval_acc'], 'train_loss': train_acc['eval_loss']}
        wandb.log(train_result)

        trainer.compute_metrics = EvalMetrics('output/bert/results', wandb.run.name, True)
        test_acc = trainer.evaluate(eval_dataset=test_dataset)
        test_result = {'test_acc': test_acc['eval_acc'], 'test_loss': test_acc['eval_loss']}
        wandb.log(test_result)

        wandb.config = args.to_dict()
        save_config(args.to_dict(), os.path.join('./output/bert/configs', wandb.run.name + '.yaml'))

        wandb.finish()
        del model, args, trainer
