import os,logging,json,random,shutil,torch,re,codecs
from collections import defaultdict
from typing import Any, Optional, Dict, Iterable, List, Tuple,NamedTuple
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
from transformers.modeling_utils import PreTrainedModel
from torch.utils.data import Dataset, BatchSampler, Sampler
from transformers import BertTokenizer
from torch.utils.data.dataloader import DataLoader
from functools import partial
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        # filename='log.txt',
                        # filemode='a'
                    )
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


class BaseArguments(object):
    def __init__(self):
        self.task_num_classes = {'0': 3, '1': 9}
        self.task_id_to_name = {'0': 'three', '1': 'nine'}
        self.task_name_to_id = {value: key for key, value in self.task_id_to_name.items()}
        self.task_data = {
            'three': {'predict': 'three_eval.txt', 'train': 'three_train.txt', "label": "three_label.txt"},
            'nine': {'predict': 'nine_eval.txt', 'train': 'nine_train.txt', "label": "nine_label.txt"},
        }

        self.model_name_or_path = ["bert-base-chinese", "hfl/chinese-roberta-wwm-ext-large", "user_data/first_stage_ckpt/checkpoint-4-180-0.2306"][0]
        self.tokenizer_dir = ["bert-base-chinese"][0]
        self.output_dir = "data/checkpoint"
        self.data_dir = "data/original_data"
        self.data_save_dir = "data/processed"
        self.max_seq_length = 510
        self.n_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.warmup_steps = 0
        self.freeze = False  # Whether to freeze bert model parameters
        self.overwrite_cache = False  # Overwrite the cached preprocessed datasets or not
        self.no_cuda = False  # 不使用cuda

        self.train_val_split_ratio = 0
        self.batch_size = 2
        self.learning_rate = 2e-5
        self.num_train_epochs = 4
        self.logging_steps = 1
        self.save_total_limit = 1


base_arguments = BaseArguments()


class PredictionOutput(NamedTuple):
    predictions: Dict[str, torch.Tensor]
    task_probs: Dict[str, torch.Tensor]
    label_ids: Optional[Dict[str, torch.Tensor]]
    metrics: Optional[Dict[str, float]]


def load_json(file_path):
    return json.load(open(file_path, 'r', encoding='utf-8'))


def load_label(file_path:str) -> Tuple[Dict,Dict]:
    content = open(file_path, mode="r", encoding="utf-8").read().splitlines()
    label2id = {}
    for line in content:
        sp = line.strip().split("\t")
        if len(sp) == 2:
            label = sp[0]
            label_id = int(sp[1])
            if label not in label2id:
                label2id[label] = label_id

    id2label = {value:key for key,value in label2id.items()}
    return label2id,id2label


def get_df(data_dir: str, data_name: str) -> list:
    data_path = os.path.join(data_dir, data_name)
    df = codecs.open(data_path, encoding="utf-8", mode="r").read().splitlines()
    return df[:]


def preprocess(args: Any, task: str):
    data_name = base_arguments.task_data[task]
    train_df = get_df(args.data_dir, data_name['train'])
    pred_df = get_df(args.data_dir, data_name['predict'])
    label2id, id2label = load_label(os.path.join(args.data_dir,data_name["label"]))
    assert len(label2id) == len(id2label) == base_arguments.task_num_classes[base_arguments.task_name_to_id[task]]
    logger.info(label2id)

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_dir)

    total_precessed_data = convert_df_to_inputs(task, tokenizer, train_df,label2id)
    train_precessed, val_precessed = train_val_split(total_precessed_data,args.train_val_split_ratio)
    predict_precessed = convert_df_to_inputs(task, tokenizer, pred_df, label2id)

    data_save_dir = os.path.join(args.data_save_dir, task)
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    print(f'Saving processed {task} data ...')
    cache_data_path = {
        'train': os.path.join(args.data_save_dir, task, 'train.pt'),
        'val': os.path.join(args.data_save_dir, task, 'val.pt'),
        'predict': os.path.join(args.data_save_dir, task, 'predict.pt')
    }
    json.dump(label2id, open(os.path.join(data_save_dir, 'label2id.json'), 'w',encoding="utf-8"))
    torch.save(train_precessed, cache_data_path['train'])
    torch.save(val_precessed, cache_data_path['val'])
    torch.save(predict_precessed, cache_data_path['predict'])

    return cache_data_path


def clean_text(text:str):
    rule = re.compile(r'(http[:0-9a-zA-Z\\/=@#$%&.?\-<>()~^{}+_]+)|(<.*?>)|(#\w{1,20}#)|(@\w{1,20}[: ])')
    text = re.sub(rule,"",text)
    return text.strip()


def convert_df_to_inputs(task: str, tokenizer: BertTokenizer, df: list, label2id: Optional[dict] = None) -> defaultdict:
    inputs = defaultdict(list)
    targets = []
    for row in tqdm(df, total=len(df), desc=f'Preprocess {task}'):
        row_list = row.split("\t")
        if len(row_list) == 3:
            text_a = clean_text(str(row_list[1]))
            if len(text_a) > 5:
                target_str = str(row_list[2])
                output_ids = tokenizer.encode_plus(text_a, add_special_tokens=True, return_token_type_ids=True,return_attention_mask=True)
                inputs['input_ids'].append(output_ids['input_ids'])
                inputs['token_type_ids'].append(output_ids['token_type_ids'])
                inputs['attention_mask'].append(output_ids['attention_mask'])
                targets.append(label2id[target_str])

    inputs['targets'] = torch.tensor(targets,dtype=torch.int64)

    assert len(list(set([len(value) for key,value in inputs.items()]))) == 1

    return inputs


def train_val_split(inputs, train_val_split_ratio):
    num_val = int(len(inputs['input_ids']) * train_val_split_ratio)
    train_data = {}
    val_data = {}
    for key, tensor in inputs.items():
        train_data[key] = tensor[num_val:]
        val_data[key] = tensor[:num_val]
    outputs = (train_data, val_data)
    return outputs


class SingleTaskDataset(Dataset):

    def __init__(self, data_path: str):
        super(SingleTaskDataset, self).__init__()
        self.data_dict = torch.load(data_path)

    def __getitem__(self, index: int) -> tuple:
        return (self.data_dict['input_ids'][index],
                self.data_dict['token_type_ids'][index],
                self.data_dict['attention_mask'][index],
                self.data_dict['targets'][index]
                )

    def __len__(self) -> int:
        return len(self.data_dict['input_ids'])


class MultiTaskDataset(Dataset):

    def __init__(self, datasets: Dict[str, SingleTaskDataset]):
        super(MultiTaskDataset, self).__init__()
        self.datasets = datasets

    def __getitem__(self, index: tuple) -> dict:
        task_id, dataset_index = index
        return {'task_id': task_id, 'data': self.datasets[task_id][dataset_index]}

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets.values())


class MultiTaskBatchSampler(BatchSampler):

    def __init__(self, datasets: MultiTaskDataset, batch_size: int, shuffle=True):
        super(MultiTaskBatchSampler, self).__init__(sampler=Sampler(datasets), batch_size=batch_size,
                                                    drop_last=False)
        self.datasets_length = {task_id: len(dataset) for task_id, dataset in datasets.datasets.items()}
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.task_indexes = []
        self.batch_indexes = {}
        self.task_num_batches = {}
        self.total_batches = 0
        self.init()

    def init(self):
        for task_id, dataset_len in self.datasets_length.items():
            num_batches = (dataset_len - 1) // self.batch_size + 1
            self.batch_indexes[task_id] = list(range(dataset_len))
            self.task_num_batches[task_id] = num_batches
            self.total_batches += num_batches
            self.task_indexes.extend([task_id] * num_batches)

    def __len__(self) -> int:
        return self.total_batches

    def __iter__(self) -> Iterable:
        batch_generator = self.get_batch_generator()
        for task_id in self.task_indexes:
            current_indexes_gen = batch_generator[task_id]
            batch = next(current_indexes_gen)
            yield [(task_id, index) for index in batch]

    def get_batch_generator(self) -> Dict[str, Iterable]:
        if self.shuffle:
            random.shuffle(self.task_indexes)
        batch_generator = {}
        for task_id, batch_indexes in self.batch_indexes.items():
            if self.shuffle:
                random.shuffle(batch_indexes)
            batch_generator[task_id] = iter([batch_indexes[i * self.batch_size: (i + 1) * self.batch_size] for i in range(self.task_num_batches[task_id])])
        return batch_generator


def collate_fn(examples: List[dict], max_seq_len: int) -> dict:
    task_ids = []
    data = []
    for example in examples:
        task_ids.append(example['task_id'])
        data.append(example['data'])

    assert (np.array(task_ids) == task_ids[0]).all(), 'batch data must belong to the same task.'

    input_ids_list, token_type_ids_list, attention_mask_list, targets_list = list(zip(*data))

    cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
    max_seq_len = min(cur_max_seq_len, max_seq_len)
    input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
    token_type_ids = torch.zeros_like(input_ids)
    attention_mask = torch.zeros_like(input_ids)
    for i in range(len(input_ids_list)):
        seq_len = min(len(input_ids_list[i]), max_seq_len)
        if seq_len <= max_seq_len:
            input_ids[i, :seq_len] = torch.tensor(input_ids_list[i][:seq_len], dtype=torch.long)
        else:
            input_ids[i, :seq_len] = torch.tensor(input_ids_list[i][:seq_len - 1] + [102], dtype=torch.long)
        token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i][:seq_len], dtype=torch.long)
        attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i][:seq_len], dtype=torch.long)
    labels = torch.tensor(targets_list, dtype=torch.long)

    return {
        'task_id': task_ids[0],
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def compute_metrics(y_true, y_pred) -> float:
    f1 = f1_score(y_true, y_pred, average='macro')
    return f1


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class LabelSmoother:
    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, logits, labels):
        log_probs = -torch.nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)
        padding_mask = labels.eq(self.ignore_index)
        labels.clamp_min_(0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss


class Trainer:
    def __init__(
            self,
            model: PreTrainedModel,
            args:BaseArguments,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            test_dataset: Optional[Dataset] = None,
            compute_metrics= None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
    ):
        self.args = args
        self.model = model

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.compute_metrics = compute_metrics
        self.optimizers = optimizers
        set_seed(self.args.seed)
        os.makedirs(self.args.output_dir, exist_ok=True)

        self.collate = partial(collate_fn, max_seq_len=self.args.max_seq_length)
        self.label_smoother = LabelSmoother()

        self.log_output_path = os.path.join(self.args.output_dir,"log")
        os.makedirs(self.log_output_path,exist_ok=True)
        self.summary_writer = SummaryWriter(log_dir=self.log_output_path)

    def init_model(self,model):
        model.to(self.args.device)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        return model

    def get_dataloader(self, dataset: Optional[Dataset] = None) -> DataLoader:
        batch_sampler = MultiTaskBatchSampler(datasets=dataset, batch_size=self.args.batch_size)
        dataloader = DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=self.collate)
        return dataloader

    def _training_step(self, inputs: Dict[str, torch.Tensor]) -> float:
        self.model.train()
        task_id = inputs.pop('task_id')
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        inputs['task_id'] = task_id
        outputs = self.model(**inputs)
        # loss = outputs[0]
        loss = self.label_smoother(outputs[1],inputs['labels'])
        if self.args.n_gpu > 1: loss = loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        return loss.item()

    def get_optimizers(self, num_training_steps: int) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        if self.optimizers is not None:
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,num_training_steps=num_training_steps)

        return optimizer, scheduler

    def train(self):
        train_dataloader = self.get_dataloader(self.train_dataset)
        t_total = int(len(train_dataloader) * self.args.num_train_epochs)
        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataloader))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Num Batch Size = %d", self.args.batch_size)
        logger.info("  Total optimization steps = %d", t_total)

        self.model = self.init_model(self.model)
        self.model.zero_grad()

        global_step = 0
        best_f1 = 0.0

        # epoch train
        for epoch in range(self.args.num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            # batch train
            for step, inputs in enumerate(train_dataloader):

                curr_step_loss = self._training_step(inputs)

                self.summary_writer.add_scalar("loss",curr_step_loss,global_step)
                self.summary_writer.add_scalar("lr",scheduler.get_last_lr()[0],global_step)

                optimizer.step()
                scheduler.step()
                self.model.zero_grad()
                global_step += 1

                # print loss when train step
                if global_step % self.args.logging_steps == 0:
                    logger.info(f"epoch={epoch},step={global_step},loss={curr_step_loss},lr={scheduler.get_last_lr()[0]}")

                    # evaluate when every epoch
                    metrics = self.evaluate(dataset=self.test_dataset,description="TestDatasetEval")
                    total_f1 = metrics["f1_avg"]

                    for key, value in metrics.items():
                        self.summary_writer.add_scalar(key, value, global_step)

                    if total_f1 > best_f1:
                        logger.info(f"Total F1 from {best_f1} improve to {total_f1}")
                        best_f1 = total_f1
                        self.save_model(epoch,global_step,metrics)

        return self.model

    def save_model(self,epoch,global_step,metrics):
        # Save model checkpoint
        best_score = metrics.pop("f1_avg")
        other_score = "-".join([key+"%.4f"%value for key,value in metrics.items()])
        file_dir = os.path.join(self.args.output_dir, f"checkpoint-{epoch}-{global_step}-{other_score}-{best_score:.6f}")

        os.makedirs(file_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", file_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(file_dir)

        self._rotate_checkpoints(self.args.output_dir)

    def _sorted_checkpoints(self, output_dir) -> List[str]:
        ordering_and_checkpoint_path = []
        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"checkpoint-*")]

        for path in glob_checkpoints:
            try:
                tup = (float(path.split("-")[-1]),path)
                ordering_and_checkpoint_path.append(tup)
            except:
                logger.info("")
        checkpoints_sorted = sorted(ordering_and_checkpoint_path,key=lambda x:x[0])
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self, output_dir) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(output_dir)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit:{}".format(checkpoint,self.args.save_total_limit))
            shutil.rmtree(checkpoint)

    def evaluate(self, dataset: Optional[Dataset] = None, description="Evaluation") -> Dict[str, float]:
        eval_dataloader = self.get_dataloader(dataset)
        output = self._prediction_loop(eval_dataloader, description=description)
        logger.info(output.metrics)
        return output.metrics

    def _prediction_loop(self, dataloader: DataLoader, description: str) -> PredictionOutput:
        model = self.model

        logger.info("***** Running %s *****", description)
        # logger.info("  Num examples = %d", len(dataloader))

        eval_losses: List[float] = []
        task_probs: Dict[str, torch.Tensor] = {}
        preds: Dict[str, torch.Tensor] = {}
        label_ids: Dict[str, torch.Tensor] = {}
        model.eval()

        for inputs in dataloader:
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

            task_id = inputs.pop('task_id')
            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)
            inputs['task_id'] = task_id

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
            pred_labels = logits.detach().argmax(dim=-1)
            if task_id not in preds:
                preds[task_id] = pred_labels
                task_probs[task_id] = probs
            else:
                task_probs[task_id] = torch.cat((task_probs[task_id], probs), dim=0)
                preds[task_id] = torch.cat((preds[task_id], pred_labels), dim=0)

            if inputs.get("labels") is not None:
                labels = inputs["labels"].detach()
                if task_id not in label_ids:
                    label_ids[task_id] = labels
                else:
                    label_ids[task_id] = torch.cat((label_ids[task_id], labels), dim=0)

        metrics = {}
        if self.compute_metrics is not None and preds and label_ids:
            for task_id, task_preds in preds.items():
                task_preds = task_preds.cpu().numpy()
                task_label_ids = label_ids[task_id].cpu().numpy()
                metrics[base_arguments.task_id_to_name[task_id]] = self.compute_metrics(task_label_ids,task_preds)
            metrics['f1_avg'] = sum(metrics.values()) / len(list(metrics.keys()))

        # if len(eval_losses) > 0:
        #     metrics["loss_avg"] = np.mean(eval_losses)

        return PredictionOutput(predictions=preds, task_probs=task_probs, label_ids=label_ids, metrics=metrics)







