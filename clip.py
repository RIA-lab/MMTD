from models import CLIPEmailModel
from Email_dataset import EDPDataset, EDPCollatorCLIP
from transformers import Trainer, TrainingArguments
from utils import metrics, save_config, SplitData, EvalMetrics
import wandb
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
fold = 5
split_data = SplitData('DATA/email_data/EDP.csv', fold)
train_df, test_df = split_data()
train_df, test_df = split_data()
train_df, test_df = split_data()


if __name__ == '__main__':
    for i in range(3, fold):
        name = 'clip-fold-' + str(i + 1)
        train_df, test_df = split_data()
        train_dataset = EDPDataset('DATA/email_data/pics', train_df)
        test_dataset = EDPDataset('DATA/email_data/pics', test_df)
        model = CLIPEmailModel.from_pretrained("openai/clip-vit-base-patch32")

        args = TrainingArguments(
            output_dir='./output/clip/checkpoints/fold' + str(i + 1),
            logging_dir='./output/clip/log',
            logging_strategy='epoch',
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=10,
            # fp16=True,
            learning_rate=2e-5,
            weight_decay=0.0,
            remove_unused_columns=False,
            save_strategy="epoch",
            evaluation_strategy="no",
            # load_best_model_at_end=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            run_name=name,
            auto_find_batch_size=False,
            overwrite_output_dir=True,
            save_total_limit=5,
            report_to=["wandb"],
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            # eval_dataset=test_dataset,
            data_collator=EDPCollatorCLIP(),
            compute_metrics=metrics,
        )

        trainer.train()
        # train_acc = trainer.evaluate(eval_dataset=train_dataset)
        # train_result = {'train_acc': train_acc['eval_acc'], 'train_loss': train_acc['eval_loss']}
        # wandb.log(train_result)

        trainer.compute_metrics = EvalMetrics('output/clip/results', name, True)
        test_acc = trainer.evaluate(eval_dataset=test_dataset)
        test_result = {'test_acc': test_acc['eval_acc'], 'test_loss': test_acc['eval_loss']}
        # wandb.log(test_result)

        # wandb.config = args.to_dict()
        # save_config(args.to_dict(), os.path.join('./output/clip/configs', wandb.run.name + '.yaml'))
        # wandb.finish()
        del model, args, trainer

