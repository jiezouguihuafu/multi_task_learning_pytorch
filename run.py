import os
from transformers import BertConfig
from utils import preprocess,SingleTaskDataset,MultiTaskDataset,compute_metrics,logger,Trainer,base_arguments
from model import BertMultiTaskModel


def get_data(data_args):
    cache_data_path = {}
    if not os.path.exists(data_args.data_save_dir) or data_args.overwrite_cache:
        for id_, task in base_arguments.task_id_to_name.items():
            task_cache_data_path = preprocess(data_args, task)
            cache_data_path[id_] = task_cache_data_path
    else:
        for id_, task_name in base_arguments.task_id_to_name.items():
            cache_data_path[id_] = {
                'train': os.path.join(data_args.data_save_dir, task_name, 'train.pt'),
                'val': os.path.join(data_args.data_save_dir, task_name, 'val.pt'),
                'predict': os.path.join(data_args.data_save_dir, task_name, 'predict.pt')
            }

    multi_task_train_dataset = {}
    multi_task_val_dataset = {}
    multi_task_predict_dataset = {}

    for id_, cache_path in cache_data_path.items():
        multi_task_train_dataset[id_] = SingleTaskDataset(cache_path['train'])
        multi_task_val_dataset[id_] = SingleTaskDataset(cache_path['val'])
        multi_task_predict_dataset[id_] = SingleTaskDataset(cache_path['predict'])

    train_dataset = MultiTaskDataset(multi_task_train_dataset)
    val_dataset = MultiTaskDataset(multi_task_val_dataset)
    predict_dataset = MultiTaskDataset(multi_task_predict_dataset)

    return train_dataset,val_dataset,predict_dataset


def train():
    train_dataset, val_dataset, predict_dataset = get_data(base_arguments)

    # 如果不是文件夹，就说明是模型名字
    if not os.path.isdir(base_arguments.model_name_or_path):
        bert_config = BertConfig.from_pretrained(base_arguments.model_name_or_path,output_hidden_states=True)
        model = BertMultiTaskModel(config=bert_config, task_num_classes=base_arguments.task_num_classes,model_path=base_arguments.model_name_or_path)
        logger.info("加载初始模型完成")
    else:
        model = BertMultiTaskModel.from_pretrained(base_arguments.model_name_or_path, task_num_classes=base_arguments.task_num_classes,model_path=base_arguments.model_name_or_path)
        logger.info("加载之前训练模型完成")

    if base_arguments.freeze:
        for p in model.bert.parameters():
            p.requires_grad = False
    trainer = Trainer(model=model, args=base_arguments, train_dataset=train_dataset, eval_dataset=val_dataset, test_dataset=predict_dataset,compute_metrics=compute_metrics)

    # Training
    model = trainer.train()
    logger.info("训练完成")


if __name__ == "__main__":
    train()

