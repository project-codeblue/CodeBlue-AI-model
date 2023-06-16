from kobert_tokenizer import KoBERTTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from torchmetrics.functional import accuracy,f1_score,auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn as nn
from data import data

LABEL_COLUMNS = ["1", "2", "3", "4", "5"]
MODEL_NAME = "skt/kobert-base-v1"
N_EPOCHS = 5
BATCH_SIZE = 32
MAX_TOKEN_COUNT = 128

# KoBERT Tokenizer 가져오기
tokenizer = KoBERTTokenizer.from_pretrained(MODEL_NAME)

# 데이터셋을 전처리해주는 클래스
class SymptomsDataset(Dataset):
    def __init__(self, dataset, tokenizer: KoBERTTokenizer, max_token_len: int = 128):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_token_len = max_token_len

    def __len__(self):
        # 데이터의 길이를 반환
        return len(self.dataset)

    def __getitem__(self, i):
        # i번째 데이터 샘플을 선택할 때 즉, dataset [i]를 쓸 때 해당 값을 인코딩하여 반환
        sentence, labels = self.dataset[i]
     
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True, # 토큰의 시작점에 [CLS], 마지막에 [SEP] 토큰을 붙임
            max_length=self.max_token_len,
            return_token_type_ids=False, # 두 개의 시퀀스 입력으로 활용할 때 0과 1로 문장의 토큰 값을 분리
            padding="max_length",
            truncation=True,
            return_attention_mask=True, # 패딩 된 부분을 알려주기 위해 사용되는 mask
            return_tensors='pt',
        )

        return dict(
            sentence = sentence,
            input_ids = encoding["input_ids"].flatten(),
            attention_mask = encoding["attention_mask"].flatten(),
            labels = torch.FloatTensor(labels)
        )
    
# DataLoader 만들기
dataset = SymptomsDataset(data, tokenizer)
sample_item = dataset[0]
sample_item.keys()

print(sample_item["sentence"])
print(sample_item["input_ids"])
print(sample_item["labels"])

# 데이터 로더에 데이터셋을 담고 next(iter.. 를 사용하여 순회 가능한 데이터를 만들어 줄 수 있음
sample_batch = next(iter(DataLoader(dataset, batch_size = 32, num_workers = 2 )))  
sample_batch["input_ids"].shape, sample_batch["attention_mask"].shape

# KoBERT 모델 가져오기
bert_model = BertModel.from_pretrained(MODEL_NAME, return_dict=True)

output = bert_model(sample_batch["input_ids"], sample_batch["attention_mask"])
print(output)

# PyTorch Lightning으로 Dataset 튜닝
class SymptomsDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, tokenizer, batch_size = 32, max_token_len = 128):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def setup(self, stage = None):
        self.train_dataset = SymptomsDataset(
            self.train_df,
            self.tokenizer,
            self.max_token_len
        )

        self.test_dataset = SymptomsDataset(
            self.test_df,
            self.tokenizer,
            self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 2
        )

    def val_dataloader(self):
        return DataLoader(
            self.batch_size,
            batch_size = self.batch_size,
            num_workers = 2
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            num_workers = 2
        )
    
# train_dataloader & test_dataloader 만들기

data_module = SymptomsDataModule(
    data, # for training
    data, # for testing
    tokenizer,
    batch_size = BATCH_SIZE
)

# PyTorch Lightning에서 trainer와 모델의 상호작용을 위해 lightning module을 만들어준다.
class SymptomsSentencesTagger(pl.LightningModule):
    def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()

    # 모델의 추론 결과를 제공하고 싶을 때 사용
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask = attention_mask)
        output = self.classifier(output.last_hidden_state[:, 0, :])
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss,outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar = True, logger = True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss,outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar =True, logger = True)
        return loss

    def on_train_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        for i, name in enumerate(LABEL_COLUMNS):
            class_roc_auc = auroc(predictions[:,i],labels[:,i])
            self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)
    
    # optimizer & scheduler 정의
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr= 2e-5)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps= self.n_warmup_steps,
            num_training_steps = self.n_training_steps
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler = dict(
            scheduler = scheduler,
            interval='step'
        )
    )

steps_per_epoch = len(data)
total_training_steps = steps_per_epoch * N_EPOCHS
warmup_steps = total_training_steps // 5

model = SymptomsSentencesTagger(
    n_classes=len(LABEL_COLUMNS),
    n_warmup_steps = warmup_steps,
    n_training_steps = total_training_steps
)

# 체크포인트
checkpoint_callback = ModelCheckpoint(
  dirpath="checkpoints",
  filename="best-checkpoint",
  save_top_k=1,
  verbose=True,
  monitor="val_loss",
  mode="min"
)

# 로깅
logger = TensorBoardLogger("lightning_logs", name="symptoms-sentences")
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)

trainer = pl.Trainer(
  logger=logger,
  callbacks=[checkpoint_callback, early_stopping_callback],
  max_epochs=N_EPOCHS,
  accelerator="gpu", 
)

trainer.fit(model, data_module)







