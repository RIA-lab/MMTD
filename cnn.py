from Email_dataset import EDPDataset, EDPPictureCollator, FeatureExtractorCNN
from transformers import Trainer, TrainingArguments
from models import CNN
from torch.optim import SGD
from utils import metrics, SplitData, save_config, EvalMetrics
import wandb
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
fold = 5
split_data = SplitData('DATA/email_data/EDP.csv', fold)

if __name__ == '__main__':
    for i in range(fold):
        wandb.init(project='MMTD')
        wandb.run.name = 'cnn-fold-' + str(i + 1)
        train_df, test_df = split_data()
        train_dataset = EDPDataset('DATA/email_data/pics', train_df)
        test_dataset = EDPDataset('DATA/email_data/pics', test_df)
        model = CNN()
        optimizer = SGD(params=model.parameters(), lr=0.01)

        args = TrainingArguments(
            output_dir='./output/cnn/checkpoints/fold' + str(i + 1),
            logging_dir='./output/cnn/log',
            logging_strategy='epoch',
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            num_train_epochs=40,
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
            data_collator=EDPPictureCollator(feature_extractor=FeatureExtractorCNN()),
            compute_metrics=metrics,
            optimizers=(optimizer, None)
        )

        trainer.train()
        train_acc = trainer.evaluate(eval_dataset=train_dataset)
        train_result = {'train_acc': train_acc['eval_acc'], 'train_loss': train_acc['eval_loss']}
        wandb.log(train_result)

        trainer.compute_metrics = EvalMetrics('output/cnn/results', wandb.run.name, True)
        test_acc = trainer.evaluate(eval_dataset=test_dataset)
        test_result = {'test_acc': test_acc['eval_acc'], 'test_loss': test_acc['eval_loss']}
        wandb.log(test_result)

        wandb.config = args.to_dict()
        save_config(args.to_dict(), os.path.join('./output/cnn/configs', wandb.run.name + '.yaml'))
        wandb.finish()
        del model, args, trainer, optimizer

