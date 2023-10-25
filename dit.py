from Email_dataset import EDPDataset, EDPPictureCollator
from transformers import BeitForImageClassification, Trainer, TrainingArguments
from utils import metrics, save_config, SplitData, EvalMetrics
import wandb
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
fold = 5
split_data = SplitData('DATA/email_data/EDP.csv', fold)


if __name__ == '__main__':
    for i in range(fold):
        wandb.init(project='MMTD')
        wandb.run.name = 'dit-fold-' + str(i + 1)
        train_df, test_df = split_data()
        train_dataset = EDPDataset('DATA/email_data/pics', train_df)
        test_dataset = EDPDataset('DATA/email_data/pics', test_df)
        model = BeitForImageClassification.from_pretrained('microsoft/dit-base')

        args = TrainingArguments(
            output_dir='./output/dit/checkpoints/fold' + str(i + 1),
            logging_dir='./output/dit/log',
            logging_strategy='epoch',
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            num_train_epochs=5,
            # fp16=True,
            learning_rate=5e-5,
            remove_unused_columns=False,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            run_name=wandb.run.name,
            auto_find_batch_size=False,
            overwrite_output_dir=True,
            save_total_limit=8,
            report_to=["wandb"],
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=EDPPictureCollator(),
            compute_metrics=metrics,
        )

        trainer.train()
        train_acc = trainer.evaluate(eval_dataset=train_dataset)
        train_result = {'train_acc': train_acc['eval_acc'], 'train_loss': train_acc['eval_loss']}
        wandb.log(train_result)

        trainer.compute_metrics = EvalMetrics('output/dit/results', wandb.run.name, True)
        test_acc = trainer.evaluate(eval_dataset=test_dataset)
        test_result = {'test_acc': test_acc['eval_acc'], 'test_loss': test_acc['eval_loss']}
        wandb.log(test_result)

        wandb.config = args.to_dict()
        save_config(args.to_dict(), os.path.join('./output/dit/configs', wandb.run.name + '.yaml'))
        wandb.finish()
        del model, args, trainer