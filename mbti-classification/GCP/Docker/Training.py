import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from transformers import AutoModel
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.optim as optim
import gc
from tqdm.auto import tqdm
from typing import Literal
import time
import psutil
import copy
import os
from google.cloud import storage

# read dataframes
#dir_path = Path("GCP/Datasets")  # local
dir_path = Path("/gcs/mbti-444713-bucket")  # GCP
data_path = dir_path / "data"
tr_user_pd = pd.read_csv(data_path / "train.csv")
val_user_pd = pd.read_csv(data_path / "validation.csv")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
torch.set_default_device(device)
print(f"Using device: {device}")

# Create PyTorch Datasets
post_feature_col = "cleaned_posts"
labels_col = "type_num"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",
                                          clean_up_tokenization_spaces=True)

class PostsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length=512,
                 post_feature_col="cleaned_posts",
                 other_feature_cols=None,
                 labels_col="type_num"):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.post_feature_col = post_feature_col
        self.other_feature_cols = other_feature_cols
        self.labels_col = labels_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row[self.post_feature_col]
        # Tokenize the text (return PyTorch tensors)
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # Squeeze out the batch dimension (from [1, seq_len] to [seq_len])
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        label = torch.tensor(row[self.labels_col]).long()

        # Engineered features converted to tensor
        if self.other_feature_cols:
            other_features = torch.tensor(row[self.other_feature_cols].values.astype(np.float32))
            return input_ids, attention_mask, other_features, label
        return input_ids, attention_mask, label

# Create datasets
train_dataset = PostsDataset(tr_user_pd, tokenizer, max_length=512,
                             # other_feature_cols=other_feature_cols,
                             post_feature_col=post_feature_col,
                             labels_col=labels_col)
val_dataset = PostsDataset(val_user_pd, tokenizer, max_length=512,
                           # other_feature_cols=other_feature_cols,
                           post_feature_col=post_feature_col,
                           labels_col=labels_col)

# Create DataLoaders
generator = torch.Generator(device=device)  # Ensure random sampling is on correct device on default
batch_size = 48

# compute the class weights for oversampling:
targets = train_dataset.df[labels_col].values
class_sample_count = np.bincount(targets)
weights = 1 / class_sample_count
sample_weights = weights[targets]  # returns an array with size of tr_user_pd

if device.type == "mps":
    torch.set_default_device("cpu")
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    torch.set_default_device(device)
else:
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

# compute class weights for balanced weighting
label_counts = train_dataset.df[labels_col].value_counts().sort_index()
total = len(train_dataset.df)
num_classes = len(label_counts)
class_weights = total / (num_classes * label_counts)
class_weights_torch = torch.tensor(class_weights.values, dtype=torch.float32)

pin_memory_value = False if device.type=="cuda" else True

# toggle between weighting and balanced sampling
sampling = False  # check

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,  # not too high on MPS
                          shuffle=True if not sampling else False,
                          sampler=sampler if sampling else None,
                          pin_memory=pin_memory_value,
                          generator=generator)  # 0 as it interferes with tokenizer

val_loader = DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        pin_memory=pin_memory_value)

# Define the model
class BertClassifier(nn.Module):
    def __init__(self, other_features_dim: int, hidden_num: int = 1, hidden_size: int = 128, num_classes: int = 16,
                 dropout_rate: float = 0.2, activation_name: Literal["silu", "relu"] = "silu",
                 use_additional_features=True, device: torch.device=device, **kwargs):
        super().__init__()

        self._init_args = {
            "other_features_dim": other_features_dim,
            "hidden_num": hidden_num,
            "hidden_size": hidden_size,
            "num_classes": num_classes,
            "dropout_rate": dropout_rate,
            "activation_name": activation_name,
            "use_additional_features": use_additional_features,
            **kwargs
        }

        self.use_additional_features = use_additional_features
        self.device = device

        # Load BERT and freeze its parameters
        self.bert = AutoModel.from_pretrained("bert-base-uncased").to(self.device)

        # freeze the bert layers initially:
        for param in self.bert.parameters():
            param.requires_grad = False

        bert_output_dim = self.bert.config.hidden_size  # typically 768 for bert-base

        # Define layers to combine BERT output and engineered features
        if self.use_additional_features:
            combined_dim = bert_output_dim + other_features_dim
        else:
            combined_dim = bert_output_dim

        # self.fc1 = nn.Linear(combined_dim, hidden_size)

        # define activation function
        activations = {"silu": nn.SiLU(), "relu": nn.ReLU()}
        self.activation = activations[activation_name]

        # set the number of hidden layers
        basic_hidden_layers = [nn.Linear(combined_dim, hidden_size), self.activation, nn.Dropout(dropout_rate)]
        additional_hidden_layers = [nn.Linear(hidden_size, hidden_size), self.activation, nn.Dropout(dropout_rate)]
        all_hidden_layers = basic_hidden_layers.copy()  # to prevent overriding basic_hidden layers in the loop
        for _ in range(hidden_num - 1):
            all_hidden_layers.extend(additional_hidden_layers)
        self.hidden_layers = nn.Sequential(*all_hidden_layers)

        # self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, other_features=None):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use the pooler output (shape: [batch_size, hidden_size])
        pooled_output = bert_outputs.pooler_output

        # Concatenate BERT embedding with additional engineered features
        if self.use_additional_features and other_features is not None:
            combined_features = torch.cat([pooled_output, other_features], dim=1)
        else:
            combined_features = pooled_output
        hidden_output = self.hidden_layers(combined_features)  # this is a block of layers
        logits = self.output_layer(hidden_output)
        return logits

    def unfreeze_layers(self, num_layers_to_unfreeze, unfreeze_pooler=True):
        """Unfreezes the pooler and last `num_layers_to_unfreeze` layers of BERT."""

        if unfreeze_pooler:
            for param in self.bert.pooler.parameters():
                param.requires_grad = True

        for param in self.bert.encoder.layer[-num_layers_to_unfreeze:].parameters():
            param.requires_grad = True

    def unpack_batch(self, batch, has_labels=True, use_additional_features=True):
        """Helper method to avoid too many indents and branches in the prediction method"""
        parts = list(batch)
        input_ids = parts[0]
        attention_mask = parts[1]

        if use_additional_features:
            other_features = parts[2]
            labels = parts[3] if has_labels else None
        else:
            other_features = None
            labels = parts[2] if has_labels else None

        return input_ids, attention_mask, other_features, labels

    def predict(self, input_ids=None, attention_mask=None, other_features=None, dataset=None, batch_size=48,
                device=device, return_labels=True, use_tqdm=True):
        if next(self.parameters()).device.type != device.type:
            self.to(device)

        self.eval()

        with torch.no_grad():
            if dataset is not None and len(dataset) <= batch_size:
                ordered_tuples = list(zip(*dataset))
                input_ids, attention_mask = [torch.stack(tup) for tup in ordered_tuples[:2]]

                if self.use_additional_features:
                    other_features = torch.stack(ordered_tuples[2])
                    if return_labels:
                        labels = torch.stack(ordered_tuples[3])
                else:
                    other_features = None
                    if return_labels:
                        labels = torch.stack(ordered_tuples[2])

                logits = self(input_ids, attention_mask, other_features)

            elif dataset is not None:
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                logits = []
                labels = [] if return_labels else None
                data_iterable = tqdm(loader, desc="Predicting", leave=False) if use_tqdm else loader

                for batch in data_iterable:
                    input_ids, attention_mask, other_features, batch_labels = self.unpack_batch(
                        batch, has_labels=return_labels, use_additional_features=self.use_additional_features
                    )

                    batch_logits = self(input_ids, attention_mask, other_features)
                    logits.append(batch_logits)
                    if return_labels:
                        labels.append(batch_labels)

                    if self.device.type == "mps":
                        torch.mps.empty_cache()
                        gc.collect()

                logits = torch.cat(logits, dim=0)
                if return_labels:
                    labels = torch.cat(labels, dim=0)

            else:
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                    attention_mask = attention_mask.unsqueeze(0)
                    if other_features is not None:
                        other_features = other_features.unsqueeze(0)

                logits = self(input_ids, attention_mask, other_features)

            probs = nn.functional.softmax(logits, dim=1)
            pred_class = probs.argmax(dim=1)

            if return_labels:
                return probs, pred_class, labels
            else:
                return probs, pred_class


# Instantiate the model with the values resulting from hypertuning and move it to the device
model = BertClassifier(other_features_dim=0, # we do not use other features
                       hidden_num=1,  #check hyper: 2, experience: 1
                       hidden_size=64,  #check hyper: 64, experience: 128
                       num_classes=16,
                       dropout_rate=0.2, #check hyper: 0.2874, experience: 0.2
                       activation_name="silu",  #check hyper: relu, experience: silu
                       use_additional_features=False)
model.to(device)

# helper function to check if we are training on the cloud
def is_cloud_env():
    """Returns True if there is a sub dir within the container that stores the bucket (and so starts with "gcs") or 
    AIP_JOB_NAME is set in the environment (what Vertex AI custom Jobs do automatically)"""
    return Path("/gcs").exists() or "AIP_JOB_NAME" in os.environ

# trainer_class.py with dynamic unfreezing
class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, lr_schedule_list: list,
                 unfreeze_layers_list: list, patience: int, patience_last_phase: int, num_epochs: int,
                 optimizer_name: Literal["adamw", "adam", "sgd"] = "adamw", device: torch.device = device,
                 class_weights: torch.tensor=None, max_batches: int = None, use_additional_features: bool = True,
                 use_dynamic_unfreezing: bool = True, warmup_epochs: int = 10, restore_best_weights_phase: bool = True,
                 num_epochs_per_unfreeze_phase: int = None, suppress_tokenizer_warning: bool = True, use_tqdm=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.starting_device = device  # to check if later a switch to another device is necessary
        self.device = device
        self.patience = patience
        self.patience_last_phase = patience_last_phase
        self.num_epochs = num_epochs
        self.lr_schedule_list = lr_schedule_list
        self.unfreeze_layers_list = unfreeze_layers_list
        self.last_index_unfreeze_layers_list = len(self.unfreeze_layers_list) - 1
        self.class_weights = class_weights
        self.num_used_batches = max_batches
        self.suppress_tokenizer_warning = suppress_tokenizer_warning  # to stop forking
        self.use_additional_features = use_additional_features  # for the ablation study
        self.use_dynamic_unfreezing = use_dynamic_unfreezing  # for hypertuning
        self.warmup_epochs = warmup_epochs
        self.restore_best_weights_phase = restore_best_weights_phase
        self.num_epochs_per_unfreeze_phase = num_epochs_per_unfreeze_phase

        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights.to(self.device) if self.class_weights is not None else None
        )

        # setting the optimizer
        self.optimizer_name = optimizer_name
        self.optimizers = {"adamw": lambda lr: optim.AdamW(self.model.parameters(), lr=lr),
                           "adam": lambda lr: optim.Adam(self.model.parameters(), lr=lr),
                           "sgd": lambda lr: optim.SGD(self.model.parameters(), lr=lr)}
        self.learning_rate = lr_schedule_list[0]  # we start with the first entry of our the lr_list
        self.optimizer = self.optimizers[self.optimizer_name](self.learning_rate)

        self.idx_unfreeze = 0  # index of our unfreeze_layers_list, starting with 0
        self.num_layers_to_unfreeze = self.unfreeze_layers_list[0]  # start with first entry of the list

        self.best_val_acc_total = 0.0
        self.best_val_acc_phase = 0.0
        self.best_model_state = None
        self.best_optimizer_state = None
        self.epochs_without_improve = 0
        self.epochs_in_current_phase = 0  # to count the number of epochs in the current unfreezing phase
        self.epoch_times = []
        self.epoch = 0
        self.cloud_run = is_cloud_env()

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "unfrozen_layers": [],
            "learning_rate": [],
            "logs": [],
            "grad_norms": [],
            "restore_best_weights": self.restore_best_weights_phase
        }
        self.setup_environment()
        self.model.to(self.device)
        self.use_tqdm = use_tqdm
        # self.log_memory("Initial state")

    def setup_environment(self):
        """check if the current environment runs on "mps" to prevent forking when tokenizing"""
        if self.suppress_tokenizer_warning and self.device.type == "mps":
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def tensors_to_device(self, batch):
        return type(batch)(tensor.to(self.device) for tensor in batch)

    def log(self, message: str, always_print: bool = True):
        # Always store in history
        # we want to know at which epoch special loggings occur:
        log = f"Epoch {self.epoch + 1}/{self.num_epochs}: {message}"
        formatted_log = f"[Trainer] {log}"
        self.history["logs"].append(formatted_log)

        # Only print based on condition (meaning that at least one must be true)
        if (always_print  # by default True must be set manually to False
                or not self.cloud_run  # False when training on cloud
                or self.epochs_in_current_phase % 5 == 0  # when conds above are both false it depends on this...
                or self.epoch == self.num_epochs - 1):  # ...and this
            if self.use_tqdm:
                tqdm.write(log)
            else:
                print(log, flush=True)

    def log_memory(self, label):
        self.log(f"\n[Memory @ {label}]")
        self.log(f"RAM used: {psutil.virtual_memory().percent}%")
        if torch.backends.mps.is_available():
            self.log(f"MPS allocated: {torch.mps.current_allocated_memory() / 1024 ** 2:.2f} MB")
            self.log(f"MPS reserved:  {torch.mps.driver_allocated_memory() / 1024 ** 2:.2f} MB")

    def free_memory(self):
        if self.device.type == "mps":
            torch.mps.empty_cache()
            gc.collect()

    def get_gradients_norm(self, model, epoch, batch_idx):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        self.history["grad_norms"].append({
            "epoch": epoch + 1,
            "batch": batch_idx,
            "norm": total_norm ** 0.5
        })

    def unfreeze_and_maybe_switch_device(self):
        self.model.unfreeze_layers(self.num_layers_to_unfreeze)

        # switch to cpu, when more than one bert layer is unfrozen and training on "mps"
        if self.num_layers_to_unfreeze > 1 and self.device.type == "mps":
            self.device = torch.device("cpu")
            self.model.to(self.device)
            self.class_weights = self.class_weights.to(self.device) if self.class_weights is not None else None
            self.log(f"Move model and data to {self.device}")

        self.learning_rate = self.lr_schedule_list[
            self.idx_unfreeze]  # comes from the training loop, when we hit a plateau. We could move it to there too
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        self.log(f"Adjusted Learning Rate to {self.learning_rate:.1e}")
        self.free_memory()

    def unfreeze_next_phase(self):
        # check if we are not already at the last phase of unfreezing layers
        if self.idx_unfreeze < self.last_index_unfreeze_layers_list:
            self.idx_unfreeze += 1
            self.num_layers_to_unfreeze = self.unfreeze_layers_list[self.idx_unfreeze]  # get next value from the list
            self.unfreeze_and_maybe_switch_device()  # Unfreeze more layers and switch to "cpu" if necessary

            self.log(f"Unfroze {self.num_layers_to_unfreeze} layers due to "
                     f"{'plateau' if self.use_dynamic_unfreezing else 'schedule'}")

            if self.restore_best_weights_phase:  # reset the parameters for the phase
                self.model.load_state_dict(self.best_model_state)
                self.optimizer.load_state_dict(self.best_optimizer_state)
                self.log("Restored weights to best epoch so far")
                self.best_val_acc_phase = self.best_val_acc_total  # to keep the patience counter equal
            else:
                self.best_val_acc_phase = 0
            self.epochs_without_improve = 0
            self.epochs_in_current_phase = 0
        else:
            self.log(f"""{'No improvement, but all layers already unfrozen:' if self.use_dynamic_unfreezing 
            else 'All scheduled layers unfrozen:'} """
                f"Stopping.")
            return True  # signal for stopping the training
        return False

    def train_one_epoch(self):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        batch_times = []

        data_iterable = tqdm(self.train_loader,
                             desc=f"Train Epoch {self.epoch + 1}",
                             leave=False,
                             disable=not self.use_tqdm)  # when false returns loader as is

        for batch_idx, batch in enumerate(data_iterable):
            # check if there was a limit on the number of batches to use
            if self.num_used_batches and batch_idx >= self.num_used_batches:
                break

            start = time.time()

            if self.starting_device.type == "mps":
                batch = self.tensors_to_device(batch)  # else tensors stay on default device

            if self.use_additional_features:
                input_ids, attention_mask, other_features, labels = batch
            else:
                input_ids, attention_mask, labels = batch
                other_features = None  # guarantees flexibility

            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, other_features)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.get_gradients_norm(self.model, self.epoch, batch_idx)  # for model inspecting
            self.optimizer.step()

            running_loss += loss.item() * input_ids.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch_times.append(time.time() - start)

            avg_loss = running_loss / total
            accuracy = correct / total

            del input_ids, attention_mask, other_features, labels, outputs, loss
            self.free_memory()  # happens only on mps device

            if self.use_tqdm:
                data_iterable.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{accuracy:.4f}")

        return avg_loss, accuracy, sum(batch_times) / len(batch_times)

    def validate(self):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        batch_times = []

        with ((torch.no_grad())):

            data_iterable = tqdm(self.val_loader,
                                 desc=f"Val Epoch {self.epoch + 1}",
                                 leave=False,
                                 disable=not self.use_tqdm)  # when false returns loader as is

            for batch_idx, batch in enumerate(data_iterable):
                # check if there was a limit on the number of batches to use
                if self.num_used_batches and batch_idx >= self.num_used_batches:
                    break

                start = time.time()

                if self.starting_device.type == "mps":
                    batch = self.tensors_to_device(batch)  # else tensors stay on default device

                if self.use_additional_features:
                    input_ids, attention_mask, other_features, labels = batch
                else:
                    input_ids, attention_mask, labels = batch
                    other_features = None  # guarantees flexibility

                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask, other_features)

                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * input_ids.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                batch_times.append(time.time() - start)

                avg_loss = running_loss / total
                accuracy = correct / total

                del input_ids, attention_mask, other_features, labels, outputs, loss
                self.free_memory()  # happens only on mps device
        return avg_loss, accuracy

    def train(self):
        if self.warmup_epochs>0:
            self.log(f"Warming up for {self.warmup_epochs} epochs:", always_print=True)

        for epoch in range(self.num_epochs):
            self.epoch = epoch

            if self.epoch <= self.warmup_epochs:
                warmup_start = 1e-6
                target_lr = self.lr_schedule_list[0]
                # we increase the learning rate with each epoch till it reaches the first value of our schedule
                self.learning_rate = warmup_start + (target_lr - warmup_start) * (self.epoch / self.warmup_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

            # give the model a bit more time to adapt on the last phase
            if self.idx_unfreeze == self.last_index_unfreeze_layers_list:
                self.patience = self.patience_last_phase

            start_time = time.time()

            train_loss, train_acc, avg_train_time = self.train_one_epoch()
            val_loss, val_acc = self.validate()

            # assign new values to history, same order as keys in history!
            new_values = (train_loss,
                          train_acc,
                          val_loss,
                          val_acc,
                          self.num_layers_to_unfreeze,
                          self.learning_rate)

            for key, new_value in zip(self.history.keys(), new_values):
                self.history[key].append(new_value)

            epoch_duration = time.time() - start_time
            self.epoch_times.append(epoch_duration)

            epoch_summary = f"Time: {epoch_duration:.2f}s | " \
                            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | " \
                            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} "

            if self.epoch < self.warmup_epochs - 1:
                self.log(epoch_summary +
                         f"LR: {self.learning_rate:.1e}", True)
            elif self.epoch == self.warmup_epochs - 1:
                self.log(epoch_summary +
                         f"LR: {self.learning_rate:.1e}\n"
                         f"Warm-up finished after epoch {self.epoch + 1}. LR schedule starts now.",
                         True)
            else:
                self.log(epoch_summary, always_print=False)  # only relevant for training on the cloud

            # Track the best accuracy during this phase but only when the warmup phase has finished
            if self.epoch >= self.warmup_epochs:
                if val_acc > self.best_val_acc_phase:
                    self.best_val_acc_phase = val_acc
                    self.epochs_without_improve = 0
                    if val_acc > self.best_val_acc_total:
                        self.best_val_acc_total = val_acc
                        # to decouple self.best_model_state from the current one
                        self.best_model_state = copy.deepcopy(self.model.state_dict())
                        self.best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
                        self.log(f"Saved best model state: train acc = {train_acc:.4f}, val acc = {val_acc:.4f}")
                else:
                    self.epochs_without_improve += 1
            else:
                self.epochs_without_improve = 0

            self.epochs_in_current_phase += 1

            # Decide whether to unfreeze next
            if self.use_dynamic_unfreezing and self.epochs_without_improve >= self.patience:  # dynamic unfreezing
                if self.unfreeze_next_phase():
                    break
            elif not self.use_dynamic_unfreezing and (
                    self.epoch + 1) % self.num_epochs_per_unfreeze_phase == 0:  # static unfreezing after fixed number of epochs
                if self.unfreeze_next_phase():
                    break

        self.log(f"Training completed after {sum(self.epoch_times) / 60:.1f} minutes - "
                 f"Best Val Accuracy: {self.best_val_acc_total:.4f}")

        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)


unfreeze_layers_list = [0, 1, 3]  # at 0 it is the defined default above

lr_schedule_list = [5e-5,  # check hyper: 4.02e-4. Fully frozen layers
                    5e-6,  # check hyper: 1.83e-5. When first unfreezing 1 layer and pooler
                    1e-6  # check hyper: 1.28e-6. When unfreezing the last 3 layers
                    ]

trainer = Trainer(model,
                  train_loader,
                  val_loader,
                  lr_schedule_list,
                  unfreeze_layers_list,
                  patience=10,  # check GCP to 8
                  patience_last_phase=10,
                  num_epochs=200,  # GCP 200 just as a fallback to keep resources low
                  optimizer_name="adamw",  # check hyper: adam, experience: adamw
                  class_weights=class_weights_torch,
                  use_additional_features=False,
                  use_dynamic_unfreezing=True,
                  warmup_epochs=10,
                  restore_best_weights_phase=False, # check
                  max_batches=None,
                  use_tqdm=False)  # GCP None
print("CWD:", Path.cwd(), "| cloud_run =", trainer.cloud_run)
trainer.train()

def save_torch_checkpoint(model, trainer, path):
    save_items = {"_init_args": model._init_args,
                  "model_state_dict": model.state_dict(),
                  "history": trainer.history,
                  "optimizer_state_dict": trainer.optimizer.state_dict()
                  }
    torch.save(save_items, path)
    print(f"Saved checkpoint for {save_items.keys()} to {path}")

# Save locally first
tmp_path = Path("/tmp/checkpoint.pth")
save_torch_checkpoint(model, trainer, tmp_path)

# Upload to GCS
client = storage.Client()
bucket = client.bucket("mbti-444713-bucket")
blob = bucket.blob("model-output/checkpoint.pth")
blob.upload_from_filename(str(tmp_path))

#PATH = Path("GCP/test_checkpoint.pth")  # local
# PATH = dir_path / "model-output" / "checkpoint.pth" # for GCP
# os.makedirs(PATH.parent, exist_ok=True)
# save_torch_checkpoint(model, trainer, PATH)

# delete before moving to GCP
# model_checkpoint = torch.load(PATH, weights_only=True)
# loaded_model = BertClassifier(**model_checkpoint["_init_args"])
# loaded_model.predict(ex_input_ids, ex_att_mask, None, return_labels=False)
# loaded_model.predict(dataset=subset)
# loaded_model.load_state_dict(model_checkpoint["model_state_dict"])
