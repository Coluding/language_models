import os
import warnings
from dataclasses import asdict
from finlm.dataset import FinLMDataset
from transformers import ElectraConfig, ElectraForMaskedLM, ElectraForPreTraining, ElectraPreTrainedModel, ElectraModel
from transformers import get_linear_schedule_with_warmup
from finlm.config import FinLMConfig, EncoderConfig
from finlm.callbacks import CallbackManager, CallbackTypes, AbstractCallback
import pandas as pd
import numpy as np
import datasets
import matplotlib.pylab as plt
import torch
from tqdm import tqdm
from torch import nn, Tensor
from typing import Optional, Any, List, Callable, Dict
import math
from torch.utils.data import DataLoader
from transformers.models.electra.modeling_electra import ElectraAttention, SequenceClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torcheval.metrics.functional import binary_precision, binary_recall
from typing import Optional, Any, List
import copy
from transformers import PreTrainedModel, RoFormerForMaskedLM, RoFormerConfig, LongformerForMaskedLM, LongformerConfig
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#########################################################################################################################
# Models for pretraining
#########################################################################################################################


def transformer_lm_factory(model_name: str, config: FinLMConfig, dataset: FinLMDataset):
    match model_name:
        case "roformer":
            model_config = RoFormerConfig(vocab_size=dataset.tokenizer.vocab_size,
                                          embedding_size=config.model_config.embedding_size,
                                          hidden_size=config.model_config.hidden_size,
                                          num_hidden_layers=config.model_config.num_hidden_layers,
                                          num_attention_heads=config.model_config.num_attention_heads,
                                          intermediate_size=config.model_config.intermediate_size,
                                          max_position_embeddings=config.model_config.max_position_embeddings,
                                          pad_token_id=dataset.tokenizer.pad_token_id,
                                          eos_token_id=dataset.tokenizer.eos_token_id,
                                          bos_token_id=dataset.tokenizer.bos_token_id, # Check what ahppens when none
                                          sep_token_id=dataset.tokenizer.sep_token_id,)

            lm = RoFormerForMaskedLM(model_config)

        case "longformer":

            if not hasattr(config.model_config, "attention_window"):
                raise ValueError("Attention window must be specified for Longformer model.")

            model_config = LongformerConfig(vocab_size=dataset.tokenizer.vocab_size,
                                            embedding_size=config.model_config.embedding_size,
                                            hidden_size=config.model_config.hidden_size,
                                            num_hidden_layers=config.model_config.num_hidden_layers,
                                            num_attention_heads=config.model_config.num_attention_heads,
                                            intermediate_size=config.model_config.intermediate_size,
                                            max_position_embeddings=config.model_config.max_position_embeddings,
                                            attention_window=config.model_config.attention_window,
                                            pad_token_id=dataset.tokenizer.pad_token_id,
                                            eos_token_id=dataset.tokenizer.eos_token_id,
                                            bos_token_id=dataset.tokenizer.bos_token_id, # Check what ahppens when none
                                            sep_token_id=dataset.tokenizer.sep_token_id,)

            lm = LongformerForMaskedLM(model_config)

        case "electra":
            model_config = ElectraConfig(vocab_size=dataset.tokenizer.vocab_size,
                                         embedding_size=config.model_config.embedding_size,
                                         hidden_size=config.model_config.hidden_size,
                                         num_hidden_layers=config.model_config.num_hidden_layers,
                                         num_attention_heads=config.model_config.num_attention_heads,
                                         max_position_embeddings=config.model_config.max_position_embeddings,
                                         eos_token_id=dataset.tokenizer.eos_token_id,
                                         bos_token_id=dataset.tokenizer.bos_token_id,
                                         sep_token_id=dataset.tokenizer.sep_token_id,
                                         pad_token_id=dataset.tokenizer.pad_token_id,)

            lm = ElectraForMaskedLM(model_config)

        case _:
            raise ValueError(f"Model name {model_name} not supported.")

    return lm, model_config


def generator_lm_factory(discriminator_config: FinLMConfig, dataset:FinLMDataset,
                         generator_size: float = 0.25, model_name: str = "electra"):
    generator_config = copy.deepcopy(discriminator_config)
    generator_config.model_config.hidden_size = int(generator_config.model_config.hidden_size * generator_size)
    generator_config.model_config.intermediate_size = int(generator_config.model_config.intermediate_size * generator_size)
    generator_config.model_config.num_hidden_layers = int(generator_config.model_config.num_hidden_layers * generator_config.model_config.generator_layer_size)
    generator_config.model_config.num_attention_heads = int(generator_config.model_config.num_attention_heads * generator_size)

    generator, generator_config = transformer_lm_factory(model_name, generator_config, dataset)

    return generator, generator_config

class PretrainLM:
    """
    A class for pretraining a language model using the configurations provided in FinLMConfig.

    This class handles the setup of the dataset, model configurations, and optimization settings 
    for pretraining a language model. It also includes utility methods for token masking and 
    directory management.

    Attributes
    ----------
    config : FinLMConfig
        Configuration object containing dataset, model, and optimization configurations.
    dataset_config : DatasetConfig
        Configuration for the dataset.
    model_config : ModelConfig
        Configuration for the model architecture.
    optimization_config : OptimizationConfig
        Configuration for the optimization settings.
    save_root_path : str
        Path where models and results will be saved.
    logger : logging.Logger
        Logger instance for logging messages related to the pretraining process.
    device : torch.device
        Device on which computations will be performed (CPU or CUDA).

    Methods
    -------
    load_dataset() -> None
        Loads the dataset based on the configuration.
    mask_tokens(inputs, mlm_probability, mask_token_id, special_token_ids, n_tokens, ignore_index=-100, hard_masking=False) -> Tuple[torch.Tensor, torch.Tensor]
        Applies masked language modeling (MLM) to the input tokens.
    _create_directory_and_return_save_path(model_type: str) -> str
        Creates a directory for saving the model and returns the path.
    _set_device() -> None
        Sets the device to CUDA if available; otherwise, defaults to CPU.
    """

    def __init__(self, config: FinLMConfig):

        """
        Initializes the PretrainLM class with the given configuration.

        Parameters
        ----------
        config : FinLMConfig
            Configuration object containing dataset, model, and optimization configurations.
        """

        self.config = config
        self.dataset_config = self.config.dataset_config
        self.model_config = self.config.model_config
        self.optimization_config = self.config.optimization_config
        self.save_root_path = config.save_models_and_results_to
        self.logger = logging.getLogger(self.__class__.__name__)
        self._set_device()
        self.callback_manager = CallbackManager()

    def load_dataset(self):

        """
        Loads the dataset based on the dataset configuration.

        This method initializes the FinLMDataset using the dataset configuration provided in 
        the FinLMConfig object.
        """

        self.dataset = FinLMDataset.from_dict(asdict(self.dataset_config))

    def load_optimization(self):

        """
        Sets up the optimizer and learning rate scheduler based on the optimization configuration.

        This method calculates the total number of training steps, initializes the AdamW optimizer,
        and configures a linear learning rate scheduler with warm-up steps.
        """

        n_sequences = 0
        for key in self.dataset.database_retrieval.keys():
            n_sequences += self.dataset.database_retrieval[key]["limit"]

        self.iteration_steps_per_epoch = int(np.ceil(n_sequences / self.dataset.batch_size))
        total_steps = self.iteration_steps_per_epoch * self.optimization_config.n_epochs
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.optimization_config.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=self.optimization_config.lr_scheduler_warm_up_steps,
                                                         num_training_steps=total_steps)

        self.callback_manager.execute_callbacks("after_optim_load", self.iteration_steps_per_epoch, self.optimizer,
                                                self.scheduler)

    @staticmethod
    def mask_tokens(inputs, mlm_probability, mask_token_id, special_token_ids, n_tokens, ignore_index=-100,
                    hard_masking=False):

        """
        Applies masked language modeling (MLM) to the input tokens.

        This method randomly masks a portion of the input tokens according to the specified 
        probability, and optionally replaces some tokens with random words or keeps them unchanged.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor containing the input token IDs.
        mlm_probability : float
            Probability of masking a token for MLM.
        mask_token_id : int
            The token ID to use for masking (typically the ID for the [MASK] token).
        special_token_ids : list[int]
            List of token IDs that should not be masked (e.g., special tokens like [CLS], [SEP]).
        n_tokens : int
            The total number of tokens in the vocabulary (used for selecting random tokens).
        ignore_index : int, optional
            The index to ignore in the loss calculation (default is -100).
        hard_masking : bool, optional
            If True, all masked tokens are replaced by the mask token; otherwise, some tokens may be 
            replaced by random tokens or left unchanged (default is False).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the masked input tensor and the corresponding labels tensor.
        """

        device = inputs.device
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
        # create special_token_mask, first set all entries to false
        special_tokens_mask = torch.full(labels.shape, False, dtype=torch.bool, device=device)
        # flag all special tokens as true
        for sp_id in special_token_ids:
            special_tokens_mask = special_tokens_mask | (inputs == sp_id)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        if ignore_index:
            labels[~masked_indices] = ignore_index  # We only compute loss on masked tokens

        if hard_masking:
            inputs[masked_indices] = mask_token_id
        else:
            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=device)).bool() & masked_indices
            inputs[indices_replaced] = mask_token_id

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(
                torch.full(labels.shape, 0.5, device=device)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(n_tokens, labels.shape, dtype=torch.long, device=device)
            inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def _create_directory_and_return_save_path(self, model_type):

        """
        Creates a directory for saving the model and returns the path.

        This method creates a new directory within the save root path for storing the model checkpoints 
        and results. The directory name is based on the model type and an incremented index.

        Parameters
        ----------
        model_type : str
            The type of model being saved (used in the directory name).

        Returns
        -------
        str
            The path to the newly created directory.
        """

        current_model_folder_paths = os.listdir(self.save_root_path)
        current_model_type_folder_names = [model for model in current_model_folder_paths if
                                           model.startswith(model_type)]
        if len(current_model_type_folder_names) > 0:
            current_model_type_index = max(
                [int(model_name.split("_")[1]) for model_name in current_model_type_folder_names])
            new_model_path = self.save_root_path + model_type + "_" + str(current_model_type_index + 1).zfill(2) + "/"
        else:
            new_model_path = self.save_root_path + model_type + "_00/"
        os.mkdir(new_model_path)
        return new_model_path

    def _set_device(self):

        """
        Sets the device to CUDA if available; otherwise, defaults to CPU.

        This method checks if a GPU is available and sets the device accordingly. If a GPU is not 
        available, a warning is logged and the device is set to CPU.
        """

        if not (torch.cuda.is_available()):
            logging.warning("GPU seems to be unavailable.")
        else:
            self.device = torch.device("cuda")

    def add_callback(self, callback: AbstractCallback):
        if not isinstance(callback, AbstractCallback):
            raise TypeError(f"Callback must be of type 'AbstractCallback'. Got {type(callback)}")
        self.callback_manager.add_callback(callback)


class PretrainMLM(PretrainLM):
    """
    A class for pretraining a Masked Language Model (MLM) using the FinLM framework.

    This class inherits from `PretrainLM` and provides specific implementations for
    preparing data, loading the model, and training the Masked Language Model (MLM).

    Attributes
    ----------
    config : FinLMConfig
        Configuration object containing dataset, model, and optimization configurations.
    dataset : FinLMDataset
        The dataset prepared for MLM training.
    model : ElectraForMaskedLM
        The Electra model configured for masked language modeling.
    optimizer : torch.optim.Optimizer
        The optimizer used for training.
    scheduler : torch.optim.lr_scheduler.LambdaLR
        The learning rate scheduler used during training.
    iteration_steps_per_epoch : int
        Number of iteration steps per training epoch.
    logger : logging.Logger
        Logger instance for logging messages related to training.
    device : torch.device
        Device on which computations will be performed (CPU or CUDA).

    Methods
    -------
    load_model() -> None
        Loads and configures the Electra model for masked language modeling.
    load_optimization() -> None
        Sets up the optimizer and learning rate scheduler based on the optimization configuration.
    prepare_data_model_optimizer() -> None
        Prepares the dataset, model, and optimizer for training.
    train() -> None
        Trains the masked language model and saves the results and model.
    """

    def __init__(self, config):

        """
        Initializes the PretrainMLM class with the given configuration.

        Parameters
        ----------
        config : FinLMConfig
            Configuration object containing dataset, model, and optimization configurations.
        """

        super().__init__(config)
        self.prepare_data_model_optimizer()

        self.training_state = None
        self.global_step = 0

        self.config_change = False
        
        self.callback_manager.execute_callbacks("after_init", self)

    def add_callback(self, callback: AbstractCallback):
        self.callback_manager.add_callback(callback)

    def load_model(self, rebuild_model_config: bool = True):

        """
        Loads and configures the Electra model for masked language modeling.

        This method initializes the Electra model using the configuration settings,
        including vocabulary size, embedding size, hidden size, and other model parameters.
        The model is then moved to the appropriate device (CPU or GPU).
        """

        if rebuild_model_config:
            self.model_config = ElectraConfig(**self.model_config.__dict__)
            self.model_config.vocab_size = self.dataset.tokenizer.vocab_size

        self.model = ElectraForMaskedLM(self.model_config)
        self.config_change = False
        self.model.to(self.device)
        self.callback_manager.execute_callbacks("after_load_modal", self.model)

    def adjust_config(self, key, value):
        """
        Adjusts the model configuration with a new value for a specified key.

        This method allows for modifying any of the model configuration parameters
        such as `embedding_size`, `hidden_size`, `num_hidden_layers`, and others.

        Args:
            key (str): The configuration parameter to be adjusted.
            value (any): The new value to be set for the specified parameter.

        Raises:
            ValueError: If the specified key does not exist in the model configuration.
        """
        self.config_change = True
        if hasattr(self.model_config, key):
            setattr(self.model_config, key, value)
            self.logger.info(f"Model configuration '{key}' updated to {value}.")
        else:
            raise ValueError(f"Key '{key}' does not exist in the model configuration.")

    def prepare_data_model_optimizer(self):

        """
        Prepares the dataset, model, and optimizer for training.

        This method calls the appropriate methods to load the dataset, load the model,
        and set up the optimizer and learning rate scheduler.
        """

        self.load_dataset()
        self.load_model()
        self.load_optimization()

    def evaluate(self, eval_metrics = dict,
                 dataset: datasets.Dataset = None,
                 metric_key_prefix: str = "eval",
                 num_steps: int = None,
                 verbose: bool = True):
        """
        Runs evaluation on n steps. Requires eval dataset.
        Returns
        -------

        """
        if dataset is None:
            dataset = self.dataset.hf_dataset

        self.logger.info("Starting with evaluation...")
        eval_steps = num_steps  #TODO: Make this variable
        self.eval_batch_size = 4  #TODO: Make this variable
        eval_inds = np.random.random_integers(0, len(dataset), eval_steps)
        sampled_dataset = dataset.select(eval_inds)
        dataloader = DataLoader(sampled_dataset, batch_size=self.eval_batch_size, shuffle=False)
        self.model.eval()
        eval_loss = 0
        eval_accuracy = 0
        n_batches = 0
        with torch.no_grad():
            runs = 0
            with tqdm(range(eval_steps), desc=f"Evaluation Loop: " ) as iterator:
                for batch in dataloader:
                    inputs, attention_mask = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)
                    inputs, labels = self.mask_tokens(
                        inputs,
                        mlm_probability=self.optimization_config.mlm_probability,
                        mask_token_id=self.dataset.mask_token_id,
                        special_token_ids=self.dataset.special_token_ids,
                        n_tokens=self.dataset.tokenizer.vocab_size
                    )

                    mlm_output = self.model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                    loss, logits = mlm_output.loss, mlm_output.logits
                    masked_ids_mask = inputs == self.dataset.tokenizer.mask_token_id
                    predictions = logits.argmax(-1)
                    eval_accuracy += (predictions[masked_ids_mask] == labels[masked_ids_mask]).float().mean()
                    eval_loss += loss.item()
                    iterator.update(self.eval_batch_size)
                    runs += 1

            final_accuracy = eval_accuracy / runs
            final_loss = eval_loss / runs

            eval_metrics[metric_key_prefix + "_accuracy"].append(final_accuracy)
            eval_metrics[metric_key_prefix + "_loss"].append(final_loss)

            self.callback_manager.execute_callbacks("after_eval", final_accuracy, final_loss, self.global_step)

            if verbose:
                self.logger.info(f"Evaluation results at step [{self.global_step}] of {metric_key_prefix} dataset: "
                                 f"MLM Accuracy: {final_accuracy:.4f} --- MLM Loss: {final_loss:.4f}")

            return eval_metrics
          

    def train_epoch(self, epoch: int, training_metrics: dict, info_step: int = 100, num_steps: int = None,
                    verbose: bool = True, tqdm_iterator: tqdm = None):

        internal_steps = 0

        if self.config_change:
            warnings.warn("You changed the model config but did not reload the model. Your changes did therefore not "
                          "affect the model architecture.")

        break_flag = True if num_steps is None else False
        if num_steps is None:
            num_steps = float("inf")

        while internal_steps < num_steps:

            for batch_id, batch in enumerate(self.dataset):
                self.callback_manager.execute_callbacks("before_batch_processing", batch_id, batch)
                inputs, attention_mask = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)

                inputs, labels = self.mask_tokens(
                    inputs,
                    mlm_probability=self.optimization_config.mlm_probability,
                    mask_token_id=self.dataset.mask_token_id,
                    special_token_ids=self.dataset.special_token_ids,
                    n_tokens=self.dataset.tokenizer.vocab_size)

                mlm_output = self.model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                mlm_loss, mlm_logits = mlm_output.loss, mlm_output.logits
                training_metrics["loss"].append(mlm_loss.item())

                self.optimizer.zero_grad()
                mlm_loss.backward()

                if self.optimization_config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                mlm_grads = [p.grad.detach().flatten() for p in self.model.parameters()]
                mlm_grad_norm = torch.cat(mlm_grads).norm()

                self.optimizer.step()
                self.scheduler.step()

                with torch.no_grad():
                    masked_ids_mask = inputs == self.dataset.tokenizer.mask_token_id
                    predictions = mlm_logits.argmax(-1)
                    mlm_accuracy = (predictions[masked_ids_mask] == labels[masked_ids_mask]).float().mean()

                training_metrics["accuracy"].append(mlm_accuracy.item())
                training_metrics["gradient_norms"].append(mlm_grad_norm.item())
                current_lr = self.scheduler.get_last_lr()[0]
                training_metrics["learning_rates"].append(current_lr)

                self.callback_manager.execute_callbacks("on_batch_end", self.global_step, mlm_loss, mlm_accuracy.item(),
                                                        mlm_grad_norm.item(), current_lr)
                self.global_step += 1
                internal_steps += 1

                if tqdm_iterator is not None:
                    tqdm_iterator.update()

                if batch_id % info_step == 0 and verbose:
                    self.logger.info(
                        f"Results after {batch_id / self.iteration_steps_per_epoch:.4%} iterations of epoch {epoch + 1}:")
                    self.logger.info(f"MLM loss: {mlm_loss.item():.4f}")
                    self.logger.info(f"Gradient norm: {mlm_grad_norm:.4f}")
                    self.logger.info(f"Current learning rate: {current_lr}")
                    self.logger.info(f"Accuracy for masking task: {mlm_accuracy.item():.4f}")
                    self.logger.info("-" * 100)

                if internal_steps > num_steps:
                    break_flag = True
                    break

            if break_flag:
                break

        self.callback_manager.execute_callbacks("after_epoch", epoch, training_metrics)

        return training_metrics

    def train(self):
        self.logger.info("Starting with training...")
        self.training_state = []
        eval_step = 1000  # TODO: make variable --- currently not used --> later for evaluation
        info_step = 100  # TODO: make variable
        training_metrics = {
            "loss": [],
            "accuracy": [],
            "gradient_norms": [],
            "learning_rates": []
        }
 
        self.callback_manager.execute_callbacks("before_training")

        self.global_step = 0  # use global steps instead of epochs and batches
        for epoch in range(self.optimization_config.n_epochs):
            self.dataset.set_dataset_offsets(epoch)
            self.dataset.prepare_data_loader()
            self.callback_manager.execute_callbacks("on_epoch_start", epoch)

            self.train_epoch(epoch, training_metrics, info_step)

        self.logger.info("...training is finished, saving results and model.")

        save_path = self._create_directory_and_return_save_path(model_type="mlm")
        training_metrics_df = pd.DataFrame(training_metrics)
        training_metrics_df.to_csv(save_path + "training_metrics.csv", index=False)
        training_metrics_df.loc[:, ["loss"]].plot()
        plt.savefig(save_path + "loss.png")
        training_metrics_df.loc[:, ["accuracy"]].plot()
        plt.savefig(save_path + "accuracy.png")
        self.model.save_pretrained(save_path + "mlm_model")
        self.config.to_json(save_path + "model_config.json")

        self.logger.info("Results and model are saved.")


class PretrainDiscriminator(PretrainLM):
    """
    A class for pretraining a discriminator model in the Electra framework using the FinLM setup.

    This class inherits from `PretrainLM` and provides specific implementations for 
    preparing data, loading the discriminator model, and training the model.

    Attributes
    ----------
    config : FinLMConfig
        Configuration object containing dataset, model, and optimization configurations.
    dataset : FinLMDataset
        The dataset prepared for discriminator training.
    model : ElectraForPreTraining
        The Electra model configured for discriminator pretraining.
    optimizer : torch.optim.Optimizer
        The optimizer used for training.
    scheduler : torch.optim.lr_scheduler.LambdaLR
        The learning rate scheduler used during training.
    iteration_steps_per_epoch : int
        Number of iteration steps per training epoch.
    logger : logging.Logger
        Logger instance for logging messages related to training.
    device : torch.device
        Device on which computations will be performed (CPU or CUDA).

    Methods
    -------
    load_model() -> None
        Loads and configures the Electra discriminator model.
    load_optimization() -> None
        Sets up the optimizer and learning rate scheduler based on the optimization configuration.
    prepare_data_model_optimizer() -> None
        Prepares the dataset, model, and optimizer for training.
    replace_masked_tokens_randomly(inputs: torch.Tensor, mlm_probability: float, mask_token_id: int, special_token_ids: list[int], n_tokens: int, hard_masking: bool = True) -> Tuple[torch.Tensor, torch.Tensor]
        Replaces masked tokens with random tokens and generates labels for discriminator training.
    train() -> None
        Trains the discriminator model and saves the results and model.
    """

    def __init__(self, config):

        """
        Initializes the PretrainDiscriminator class with the given configuration.

        Parameters
        ----------
        config : FinLMConfig
            Configuration object containing dataset, model, and optimization configurations.
        """

        super().__init__(config)
        self.callback_manager.execute_callbacks("after_init", self)
        self.prepare_data_model_optimizer()

    def load_model(self):

        """
        Loads and configures the Electra discriminator model.

        This method initializes the Electra model for discriminator pretraining using the configuration settings, 
        including vocabulary size, embedding size, hidden size, and other model parameters. The model is then moved 
        to the appropriate device (CPU or GPU).
        """

        self.model_config = ElectraConfig(
            vocab_size=self.dataset.tokenizer.vocab_size,
            embedding_size=self.model_config.embedding_size,
            hidden_size=self.model_config.hidden_size,
            num_hidden_layers=self.model_config.num_hidden_layers,
            num_attention_heads=self.model_config.num_attention_heads,
            max_position_embeddings=self.model_config.max_position_embeddings
        )

        self.model = ElectraForPreTraining(self.model_config)
        self.model.to(self.device)

        self.callback_manager.execute_callbacks("after_load_modal", self.model)


    def prepare_data_model_optimizer(self):

        """
        Prepares the dataset, model, and optimizer for training.

        This method calls the appropriate methods to load the dataset, load the model, 
        and set up the optimizer and learning rate scheduler.
        """

        self.load_dataset()
        self.load_optimization()

    def replace_masked_tokens_randomly(self, inputs, mlm_probability, mask_token_id, special_token_ids, n_tokens,
                                       hard_masking=True):

        """
        Replaces masked tokens with random tokens and generates labels for discriminator training.

        This method first applies masked language modeling (MLM) to the input tokens. It then replaces 
        the masked tokens with random tokens and generates labels indicating whether a token has been 
        replaced (1) or not (0).

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor containing the input token IDs.
        mlm_probability : float
            Probability of masking a token for MLM.
        mask_token_id : int
            The token ID to use for masking (typically the ID for the [MASK] token).
        special_token_ids : list[int]
            List of token IDs that should not be masked (e.g., special tokens like [CLS], [SEP]).
        n_tokens : int
            The total number of tokens in the vocabulary (used for selecting random tokens).
        hard_masking : bool, optional
            If True, all masked tokens are replaced by the mask token; otherwise, some tokens may be 
            replaced by random tokens or left unchanged (default is True).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the corrupted input tensor and the corresponding labels tensor.
        """

        device = inputs.device
        masked_inputs, original_inputs = self.mask_tokens(
            inputs=inputs,
            mlm_probability=mlm_probability,
            mask_token_id=mask_token_id,
            special_token_ids=special_token_ids,
            n_tokens=n_tokens,
            ignore_index=None,
            hard_masking=hard_masking
        )

        masked_indices = torch.where(masked_inputs == mask_token_id, True, False)
        random_words = torch.randint(n_tokens, original_inputs.shape, dtype=torch.long, device=device)
        corrupted_inputs = original_inputs.clone()
        corrupted_inputs[masked_indices] = random_words[masked_indices]
        labels = torch.full(corrupted_inputs.shape, False, dtype=torch.bool, device=device)
        labels[masked_indices] = original_inputs[masked_indices] != corrupted_inputs[masked_indices]

        return corrupted_inputs, labels.float()

    def train(self):

        """
        Trains the discriminator model and saves the results and model.

        This method handles the training loop, including replacing masked tokens, calculating 
        the discriminator loss, updating model parameters, and logging training metrics. After 
        training is complete, it saves the model, training metrics, and plots of the loss, accuracy, 
        precision, and recall.
        """

        self.logger.info("Starting with training...")

        training_metrics = {}
        training_metrics["loss"] = []
        training_metrics["accuracy"] = []
        training_metrics["precision"] = []
        training_metrics["recall"] = []
        training_metrics["gradient_norms"] = []
        training_metrics["learning_rates"] = []


        for epoch in range(self.optimization_config.n_epochs):

            # update the offset for database retrieval, epoch = 0 -> offset = 0, epoch = 1 -> offset = 1 * limit, epoch = 2 -> offset = 2 * limit, ...    
            self.dataset.set_dataset_offsets(epoch)
            self.dataset.prepare_data_loader()
            self.callback_manager.execute_callbacks("on_epoch_start", epoch) #TODO: What to include here?

            for batch_id, batch in enumerate(self.dataset):
                self.callback_manager.execute_callbacks("before_batch_processing", batch_id, batch)
                inputs, attention_mask = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)

                inputs, labels = self.replace_masked_tokens_randomly(
                    inputs,
                    mlm_probability=self.optimization_config.mlm_probability,
                    mask_token_id=self.dataset.mask_token_id,
                    special_token_ids=self.dataset.special_token_ids,
                    n_tokens=self.dataset.tokenizer.vocab_size,
                    hard_masking=True
                )

                discriminator_output = self.model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                discriminator_loss, discriminator_logits = discriminator_output.loss, discriminator_output.logits
                training_metrics["loss"].append(discriminator_loss.item())

                # gradient determination and update
                self.optimizer.zero_grad()

                # determine gradients
                discriminator_loss.backward()

                if self.optimization_config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # determine gradient norms, equal to one if use_gradient_clipping is set to True
                discriminator_grads = [p.grad.detach().flatten() for p in self.model.parameters()]
                discriminator_grad_norm = torch.cat(discriminator_grads).norm()

                # update parameters        
                self.optimizer.step()
                # update learning rate
                self.scheduler.step()

                # determine accuracy metrics, (maybe check for correctness later, has been implemented quickly;))
                with torch.no_grad():
                    active_loss = attention_mask == 1
                    active_logits = discriminator_logits[active_loss]
                    active_predictions = (torch.sign(active_logits) + 1.0) * 0.5
                    active_labels = labels[active_loss]

                    discriminator_accuracy = (active_predictions == active_labels).float().mean()
                    discriminator_precision = binary_precision(active_predictions.long(), active_labels.long())
                    discriminator_recall = binary_recall(active_predictions.long(), active_labels.long())

                training_metrics["accuracy"].append(discriminator_accuracy.item())
                training_metrics["precision"].append(discriminator_precision.item())
                training_metrics["recall"].append(discriminator_recall.item())
                training_metrics["gradient_norms"].append(discriminator_grad_norm.item())
                current_lr = self.scheduler.get_last_lr()[0]
                training_metrics["learning_rates"].append(current_lr)

                if batch_id % 100 == 0:
                    self.logger.info(
                        f"Results after {batch_id / self.iteration_steps_per_epoch:.4%} iterations of epoch {epoch + 1}:")
                    self.logger.info(f"Discriminator loss: {discriminator_loss.item():.4f}")
                    self.logger.info(f"Gradient norm: {discriminator_grad_norm:.4f}")
                    self.logger.info(f"Current learning rate: {current_lr}")
                    self.logger.info(f"Accuracy for replacement task: {discriminator_accuracy.item():.4f}")
                    self.logger.info(f"Precision for replacement task: {discriminator_precision.item():.4f}")
                    self.logger.info(f"Recall for replacement task: {discriminator_recall.item():.4f}")

                    self.logger.info("-"*100)

                self.callback_manager.execute_callbacks("on_batch_end", self.global_step, discriminator_loss,
                                                        discriminator_accuracy.item(),
                                                        discriminator_grad_norm.item(), current_lr)

                self.global_step += 1

            self.callback_manager.execute_callbacks("after_epoch", epoch, training_metrics, self.model)

        self.logger.info("...training is finished, saving results and model.")

        save_path = self._create_directory_and_return_save_path(model_type="discriminator")
        training_metrics_df = pd.DataFrame(training_metrics)
        training_metrics_df.to_csv(save_path + "training_metrics.csv", index=False)
        training_metrics_df.loc[:, ["loss"]].plot()
        plt.savefig(save_path + "loss.png")
        training_metrics_df.loc[:, ["accuracy", "precision", "recall"]].plot(subplots=True)
        plt.savefig(save_path + "accuracy.png")
        self.model.save_pretrained(save_path + "discriminator_model")
        self.config.to_json(save_path + "model_config.json")

        self.logger.info("Results and model are saved.")


class PretrainElectra(PretrainLM):
    """
    A class for pretraining the Electra model using the FinLM setup.

    This class inherits from `PretrainLM` and provides specific implementations for 
    preparing data, loading both the generator and discriminator models, and training 
    the Electra model, which includes both components.

    Attributes
    ----------
    config : FinLMConfig
        Configuration object containing dataset, model, and optimization configurations.
    dataset : FinLMDataset
        The dataset prepared for Electra model training.
    generator : ElectraForMaskedLM
        The generator model in the Electra framework configured for masked language modeling.
    discriminator : ElectraForPreTraining
        The discriminator model in the Electra framework configured for identifying replaced tokens.
    optimizer : torch.optim.Optimizer
        The optimizer used for training.
    scheduler : torch.optim.lr_scheduler.LambdaLR
        The learning rate scheduler used during training.
    iteration_steps_per_epoch : int
        Number of iteration steps per training epoch.
    logger : logging.Logger
        Logger instance for logging messages related to training.
    device : torch.device
        Device on which computations will be performed (CPU or CUDA).

    Methods
    -------
    load_model() -> None
        Loads and configures the Electra generator and discriminator models.
    load_optimization() -> None
        Sets up the optimizer and learning rate scheduler based on the optimization configuration.
    prepare_data_model_optimizer() -> None
        Prepares the dataset, models, and optimizer for training.
    replace_masked_tokens_from_generator(masked_inputs: torch.Tensor, original_inputs: torch.Tensor, logits: torch.Tensor, special_mask_id: int, discriminator_sampling: str = "multinomial") -> Tuple[torch.Tensor, torch.Tensor]
        Replaces masked tokens with tokens sampled from the generator and generates labels for discriminator training.
    train() -> None
        Trains the Electra model, which includes both the generator and discriminator, and saves the results and models.
    """

    def __init__(self, config):

        """
        Initializes the PretrainElectra class with the given configuration.

        Parameters
        ----------
        config : FinLMConfig
            Configuration object containing dataset, model, and optimization configurations.
        """

        super().__init__(config)
        self.prepare_data_model_optimizer()

        self.global_step = 0

    def load_model(self):

        """
        Loads and configures the Electra generator and discriminator models.

        This method initializes the Electra generator and discriminator models using the 
        configuration settings, including vocabulary size, embedding size, hidden size, 
        and other model parameters. The models are then moved to the appropriate device (CPU or GPU).
        """

        self.generator_model_config = ElectraConfig(
            vocab_size=self.dataset.tokenizer.vocab_size,
            embedding_size=self.model_config.embedding_size,
            hidden_size=int(self.model_config.hidden_size * self.model_config.generator_size),
            intermediate_size=int(self.model_config.intermediate_size * self.model_config.generator_size),
            num_hidden_layers=int(self.model_config.num_hidden_layers * self.model_config.generator_layer_size),
            num_attention_heads=int(self.model_config.num_attention_heads * self.model_config.generator_size),
            max_position_embeddings=self.model_config.max_position_embeddings
        )

        self.discriminator_model_config = ElectraConfig(
            vocab_size=self.dataset.tokenizer.vocab_size,
            embedding_size=self.model_config.embedding_size,
            hidden_size=self.model_config.hidden_size,
            num_hidden_layers=self.model_config.num_hidden_layers,
            num_attention_heads=self.model_config.num_attention_heads,
            max_position_embeddings=self.model_config.max_position_embeddings
        )

        self.generator = ElectraForMaskedLM(self.generator_model_config)
        self.discriminator = ElectraForPreTraining(self.discriminator_model_config)
        # tie word and position embeddings
        self.generator.electra.embeddings.word_embeddings = self.discriminator.electra.embeddings.word_embeddings
        self.generator.electra.embeddings.position_embeddings = self.discriminator.electra.embeddings.position_embeddings
        # add to device
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.callback_manager.execute_callbacks("after_load_modal", self.generator, self.discriminator)

    def load_optimization(self):

        """
        Sets up the optimizer and learning rate scheduler based on the optimization configuration.

        This method identifies the trainable parameters, ensuring that the word and position embeddings 
        are not duplicated. It then calculates the total number of training steps, initializes the AdamW 
        optimizer, and configures a linear learning rate scheduler with warm-up steps.
        """

        # identify trainable parameters without duplicating the embedding and position parameters
        self.model_parameters = []
        # generator
        for name, params in self.discriminator.named_parameters():
            self.model_parameters.append(params)
        # discriminator
        for name, params in self.generator.named_parameters():
            if name.endswith("word_embeddings.weight") | name.endswith("position_embeddings.weight"):
                continue
            else:
                self.model_parameters.append(params)

        n_sequences = 0
        for key in self.dataset.database_retrieval.keys():
            n_sequences += self.dataset.database_retrieval[key]["limit"]
        self.iteration_steps_per_epoch = int(np.ceil(n_sequences / self.dataset.batch_size))
        total_steps = self.iteration_steps_per_epoch * self.optimization_config.n_epochs
        self.optimizer = torch.optim.AdamW(self.model_parameters, lr=self.optimization_config.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=self.optimization_config.lr_scheduler_warm_up_steps,
                                                         num_training_steps=total_steps)

        self.callback_manager.execute_callbacks("after_optim_load", self.iteration_steps_per_epoch, self.optimizer,
                                                self.scheduler)

    def prepare_data_model_optimizer(self):

        """
        Prepares the dataset, models, and optimizer for training.

        This method calls the appropriate methods to load the dataset, load the generator and 
        discriminator models, and set up the optimizer and learning rate scheduler.
        """

        self.load_dataset()
        self.load_model()
        self.load_optimization()

    @staticmethod
    def replace_masked_tokens_from_generator(masked_inputs, original_inputs, logits, special_mask_id,
                                             discriminator_sampling="gumbel_softmax"):

        """
        Replaces masked tokens with tokens sampled from the generator and generates labels for discriminator training.

        This method uses the generator's output logits to replace masked tokens in the input. It generates labels 
        indicating whether a token has been replaced and whether the replacement matches the original token.

        Parameters
        ----------
        masked_inputs : torch.Tensor
            Tensor containing the masked input token IDs.
        original_inputs : torch.Tensor
            Tensor containing the original input token IDs before masking.
        logits : torch.Tensor
            Logits output by the generator model.
        special_mask_id : int
            The token ID used for masking (typically the ID for the [MASK] token).
        discriminator_sampling : str, optional
            The sampling strategy for selecting replacement tokens, either "multinomial" or another strategy like "aggressive" or "gumbel_softmax" (default is "gumbel_softmax").

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the discriminator inputs (with replaced tokens) and the corresponding labels tensor.
        """

        device = masked_inputs.device
        discriminator_inputs = masked_inputs.clone()
        mask_indices = masked_inputs == special_mask_id

        if discriminator_sampling == "aggressive":
            sampled_ids = logits[mask_indices].argmax(-1)
        elif discriminator_sampling == "gumbel_softmax":
            sampled_ids = torch.nn.functional.gumbel_softmax(logits[mask_indices], hard=False).argmax(-1)
        else:
            sampled_ids = torch.multinomial(torch.nn.functional.softmax(logits[mask_indices], dim=-1), 1).squeeze()

        discriminator_inputs[mask_indices] = sampled_ids
        # initialize discriminator labels with False
        discriminator_labels = torch.full(masked_inputs.shape, False, dtype=torch.bool, device=device)
        # replace False with True if an id is sampled and not the same as the original one
        discriminator_labels[mask_indices] = discriminator_inputs[mask_indices] != original_inputs[mask_indices]
        # convert to float 
        discriminator_labels = discriminator_labels.float()

        return discriminator_inputs, discriminator_labels

    def train(self):

        """
        Trains the Electra model, which includes both the generator and discriminator, and saves the results and models.

        This method handles the training loop, including masking input tokens, generating replacements using the generator, 
        training the discriminator on identifying the replaced tokens, calculating losses, updating model parameters, and 
        logging training metrics. After training is complete, it saves the models, training metrics, and plots of the loss, 
        accuracy, precision, and recall.
        """

        self.logger.info("Starting with training...")
        training_metrics = {}
        training_metrics["loss"] = []
        training_metrics["mlm_loss"] = []
        training_metrics["discriminator_loss"] = []
        training_metrics["mlm_accuracy"] = []
        training_metrics["discriminator_accuracy"] = []
        training_metrics["discriminator_precision"] = []
        training_metrics["discriminator_recall"] = []
        training_metrics["gradient_norm"] = []
        training_metrics["learning_rates"] = []

        for epoch in range(self.optimization_config.n_epochs): 
            
            # update the offset for database retrieval, epoch = 0 -> offset = 0, epoch = 1 -> offset = 1 * limit, epoch = 2 -> offset = 2 * limit, ...    
            self.dataset.set_dataset_offsets(epoch)
            self.dataset.prepare_data_loader()
            self.callback_manager.execute_callbacks("on_epoch_start", epoch) #TODO: What to include here?


            for batch_id, batch in enumerate(self.dataset):
                self.callback_manager.execute_callbacks("before_batch_processing", batch_id, batch)
                inputs, attention_mask = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)

                original_inputs = inputs.clone()
                generator_inputs, generator_labels = self.mask_tokens(
                    inputs,
                    mlm_probability=self.optimization_config.mlm_probability,
                    mask_token_id=self.dataset.mask_token_id,
                    special_token_ids=self.dataset.special_token_ids,
                    n_tokens=self.dataset.tokenizer.vocab_size)

                mlm_output = self.generator(input_ids=generator_inputs, attention_mask=attention_mask,
                                            labels=generator_labels)
                mlm_loss, mlm_logits = mlm_output.loss, mlm_output.logits

                sampling_logits = mlm_logits.detach()
                discriminator_inputs, discriminator_labels = self.replace_masked_tokens_from_generator(
                    masked_inputs=generator_inputs,
                    original_inputs=original_inputs,
                    logits=sampling_logits,
                    special_mask_id=self.dataset.tokenizer.mask_token_id,
                    discriminator_sampling=self.optimization_config.discriminator_sampling
                )

                discriminator_output = self.discriminator(input_ids=discriminator_inputs, attention_mask=attention_mask,
                                                          labels=discriminator_labels)
                discriminator_loss, discriminator_logits = discriminator_output.loss, discriminator_output.logits

                loss = mlm_loss + self.optimization_config.discriminator_weight * discriminator_loss

                training_metrics["loss"].append(loss.item())
                training_metrics["mlm_loss"].append(mlm_loss.item())
                training_metrics["discriminator_loss"].append(discriminator_loss.item())

                # gradient determination and update
                self.optimizer.zero_grad()

                # determine gradients
                loss.backward()

                if self.optimization_config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model_parameters, max_norm=1.0)

                # determine gradient norms, equal to one if use_gradient_clipping is set to True
                grads = [p.grad.detach().flatten() for p in self.model_parameters]
                grad_norm = torch.cat(grads).norm()

                # update parameters
                self.optimizer.step()
                # update learning rate
                self.scheduler.step()

                # determine accuracy metrics, (maybe check for correctness later, has been implemented quickly;))
                with torch.no_grad():
                    # mask to identify ids which have been masked before
                    masked_ids_mask = inputs == self.dataset.tokenizer.mask_token_id
                    predictions = mlm_logits.argmax(-1)
                    mlm_accuracy = (predictions[masked_ids_mask] == generator_labels[masked_ids_mask]).float().mean()
                    active_loss = attention_mask == 1
                    active_logits = discriminator_logits[active_loss]
                    active_predictions = (torch.sign(active_logits) + 1.0) * 0.5
                    active_labels = discriminator_labels[active_loss]
                    discriminator_accuracy = (active_predictions == active_labels).float().mean()
                    discriminator_precision = binary_precision(active_predictions.long(), active_labels.long())
                    discriminator_recall = binary_recall(active_predictions.long(), active_labels.long())

                training_metrics["mlm_accuracy"].append(mlm_accuracy.item())
                training_metrics["discriminator_accuracy"].append(discriminator_accuracy.item())
                training_metrics["discriminator_precision"].append(discriminator_precision.item())
                training_metrics["discriminator_recall"].append(discriminator_recall.item())

                training_metrics["gradient_norm"] = grad_norm.item()
                current_lr = self.scheduler.get_last_lr()[0]
                training_metrics["learning_rates"].append(current_lr)

                if batch_id % 100 == 0:
                    self.logger.info(
                        f"Results after {batch_id / self.iteration_steps_per_epoch:.4%} iterations of epoch {epoch + 1}:")
                    self.logger.info(f"Loss: {loss.item():.4f}")
                    self.logger.info(f"MLM Loss: {mlm_loss.item():.4f}")
                    self.logger.info(f"Discriminator Loss: {discriminator_loss.item():.4f}")
                    self.logger.info(f"Gradient norm: {grad_norm:.4f}")
                    self.logger.info(f"Current learning rate: {current_lr}")
                    self.logger.info(f"Accuracy for masking task: {mlm_accuracy.item():.4f}")
                    self.logger.info(f"Accuracy for replacement task: {discriminator_accuracy.item():.4f}")
                    self.logger.info(f"Precision for replacement task: {discriminator_precision.item():.4f}")
                    self.logger.info(f"Recall for replacement task: {discriminator_recall.item():.4f}")
                    self.logger.info("-"*100)

                self.callback_manager.execute_callbacks("on_batch_end", self.global_step, mlm_loss, mlm_accuracy,
                                                        discriminator_loss, discriminator_precision,
                                                        discriminator_recall, current_lr)
                self.global_step += 1

            self.callback_manager.execute_callbacks("after_epoch", epoch, training_metrics, self.generator, self.discriminator)

        self.logger.info("...training is finished, saving results and model.")

        training_metrics_df = pd.DataFrame(training_metrics)

        save_path = self._create_directory_and_return_save_path(model_type="electra")
        # create a function for making an output directory which creates it and saves the csv and model
        training_metrics_df.to_csv(save_path + "training_metrics.csv", index=False)
        training_metrics_df.loc[:, ["loss", "mlm_loss", "discriminator_loss"]].plot(subplots=True)
        plt.savefig(save_path + "loss.png")
        training_metrics_df.loc[:,
        ["mlm_accuracy", "discriminator_accuracy", "discriminator_precision", "discriminator_recall"]].plot(
            subplots=True)
        plt.savefig(save_path + "accuracy.png")
        self.generator.save_pretrained(save_path + "mlm_model")
        self.discriminator.save_pretrained(save_path + "discriminator_model")
        self.config.to_json(save_path + "model_config.json")

        self.logger.info("Results and model are saved.")


#########################################################################################################################
# Models for finetuning
#########################################################################################################################

#TODO:
#   a model for comparison which simply adds a dense layer on top of the embeddings and uses an output projection afterwards
#   experiment with extensions of the attention model: add a dense layer after attention, maybe replace mean aggreagation by 2D convolution


class ElectraDocumentClassification(ElectraPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config
        self.num_labels = self.config.num_labels

        self.electra = ElectraModel(self.config)
        self.attention_config = copy.deepcopy(self.config)
        if "num_sequence_attention_heads" in kwargs:
            self.attention_config.num_attention_heads = kwargs["num_sequence_attention_heads"]
        self.attention = ElectraAttention(self.attention_config)
        aggregation_head_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out_projection = nn.Linear(self.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(aggregation_head_dropout)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            sequence_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None
    ):

        original_normalized_shape = input_ids.size()
        flattened_input_ids = input_ids.view(-1, original_normalized_shape[-1])
        flattened_attention_mask = attention_mask.view(-1, original_normalized_shape[-1])

        hidden_states = self.electra(input_ids=flattened_input_ids, attention_mask=flattened_attention_mask)
        encoder_embeddings = hidden_states.last_hidden_state
        sequence_embeddings = encoder_embeddings[:, 0, :]
        document_sequences_shape = original_normalized_shape[:2] + torch.Size([self.config.hidden_size])
        sequence_embeddings = sequence_embeddings.view(document_sequences_shape)

        # the shape gets adjusted internally if we write it like this
        extended_sequence_mask = self.get_extended_attention_mask(sequence_mask, sequence_mask.shape)

        # in my understanding the output of this layer is a number_of_sequences x hidden size representation
        # where the padded sequences are ignored, for instance if a attention probability matrix looks like
        # att_probs         value_repr.
        # 0.5, 0.5, 0       v_11, v_12
        # 0.5, 0.5, 0       v_21, v_22
        # 0.5, 0.5, 0       v_31, v_32
        #
        # a matrix of size n_sequences times hidden_dim is generated where only values of the first two rows are used
        # accordingly, we can average attention states in the next step and project them to the desired output dimension

        attention_embeddings, attention_probs = self.attention(hidden_states=sequence_embeddings,
                                                               attention_mask=extended_sequence_mask,
                                                               output_attentions=True)
        aggregated_attention_embeddings = attention_embeddings.mean(dim=1)

        # do we need this or another form of normalization, because of the averaging from the previous step
        aggregated_attention_embeddings = self.LayerNorm(aggregated_attention_embeddings)
        aggregated_attention_embeddings = self.dropout(aggregated_attention_embeddings)
        logits = self.out_projection(aggregated_attention_embeddings)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=attention_probs,
        )

class PretrainMLM2(PretrainLM):
    """
    A class for pretraining a Masked Language Model (MLM) using the FinLM framework.

    This class inherits from `PretrainLM` and provides specific implementations for
    preparing data, loading the model, and training the Masked Language Model (MLM).

    Attributes
    ----------
    config : FinLMConfig
        Configuration object containing dataset, model, and optimization configurations.
    dataset : FinLMDataset
        The dataset prepared for MLM training.
    model : ElectraForMaskedLM
        The Electra model configured for masked language modeling.
    optimizer : torch.optim.Optimizer
        The optimizer used for training.
    scheduler : torch.optim.lr_scheduler.LambdaLR
        The learning rate scheduler used during training.
    iteration_steps_per_epoch : int
        Number of iteration steps per training epoch.
    logger : logging.Logger
        Logger instance for logging messages related to training.
    device : torch.device
        Device on which computations will be performed (CPU or CUDA).

    Methods
    -------
    load_model() -> None
        Loads and configures the Electra model for masked language modeling.
    load_optimization() -> None
        Sets up the optimizer and learning rate scheduler based on the optimization configuration.
    prepare_data_model_optimizer() -> None
        Prepares the dataset, model, and optimizer for training.
    train() -> None
        Trains the masked language model and saves the results and model.
    """

    def __init__(self, config):

        """
        Initializes the PretrainMLM class with the given configuration.

        Parameters
        ----------
        config : FinLMConfig
            Configuration object containing dataset, model, and optimization configurations.
        """

        super().__init__(config)
        self.prepare_data_model_optimizer()

        self.training_state = None
        self.global_step = 0

        self.config_change = False

        self.callback_manager.execute_callbacks("after_init", self)

    def add_callback(self, callback: AbstractCallback):
        self.callback_manager.add_callback(callback)

    def load_model(self, rebuild_model_config: bool = True):

        """
        Loads and configures the Electra model for masked language modeling.

        This method initializes the Electra model using the configuration settings,
        including vocabulary size, embedding size, hidden size, and other model parameters.
        The model is then moved to the appropriate device (CPU or GPU).
        """

        if rebuild_model_config:
            self.model_config.vocab_size = self.dataset.tokenizer.vocab_size

        model, model_config = transformer_lm_factory(self.config.model_config.model_name, self.config, self.dataset)
        self.model = model
        self.config_change = False
        self.model.to(self.device)
        self.callback_manager.execute_callbacks("after_load_modal", self.model)

    def adjust_config(self, key, value):
        """
        Adjusts the model configuration with a new value for a specified key.

        This method allows for modifying any of the model configuration parameters
        such as `embedding_size`, `hidden_size`, `num_hidden_layers`, and others.

        Args:
            key (str): The configuration parameter to be adjusted.
            value (any): The new value to be set for the specified parameter.

        Raises:
            ValueError: If the specified key does not exist in the model configuration.
        """
        self.config_change = True
        if hasattr(self.model_config, key):
            setattr(self.model_config, key, value)
            self.logger.info(f"Model configuration '{key}' updated to {value}.")
        else:
            raise ValueError(f"Key '{key}' does not exist in the model configuration.")

    def prepare_data_model_optimizer(self):

        """
        Prepares the dataset, model, and optimizer for training.

        This method calls the appropriate methods to load the dataset, load the model,
        and set up the optimizer and learning rate scheduler.
        """

        self.load_dataset()
        self.load_model()
        self.load_optimization()

    def evaluate(self, eval_metrics=dict,
                 dataset: datasets.Dataset = None,
                 metric_key_prefix: str = "eval",
                 num_steps: int = None,
                 verbose: bool = True):
        """
        Runs evaluation on n steps. Requires eval dataset.
        Returns
        -------

        """
        if dataset is None:
            dataset = self.dataset.hf_dataset

        self.logger.info("Starting with evaluation...")
        eval_steps = num_steps  # TODO: Make this variable
        self.eval_batch_size = 4  # TODO: Make this variable
        eval_inds = np.random.random_integers(0, len(dataset), eval_steps)
        sampled_dataset = dataset.select(eval_inds)
        dataloader = DataLoader(sampled_dataset, batch_size=self.eval_batch_size, shuffle=False)
        self.model.eval()
        eval_loss = 0
        eval_accuracy = 0
        n_batches = 0
        with torch.no_grad():
            runs = 0
            with tqdm(range(eval_steps), desc=f"Evaluation Loop: ") as iterator:
                for batch in dataloader:
                    inputs, attention_mask = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)
                    inputs, labels = self.mask_tokens(
                        inputs,
                        mlm_probability=self.optimization_config.mlm_probability,
                        mask_token_id=self.dataset.mask_token_id,
                        special_token_ids=self.dataset.special_token_ids,
                        n_tokens=self.dataset.tokenizer.vocab_size
                    )

                    mlm_output = self.model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                    loss, logits = mlm_output.loss, mlm_output.logits
                    masked_ids_mask = inputs == self.dataset.tokenizer.mask_token_id
                    predictions = logits.argmax(-1)
                    eval_accuracy += (predictions[masked_ids_mask] == labels[masked_ids_mask]).float().mean()
                    eval_loss += loss.item()
                    iterator.update(self.eval_batch_size)
                    runs += 1

            final_accuracy = eval_accuracy / runs
            final_loss = eval_loss / runs

            eval_metrics[metric_key_prefix + "_accuracy"].append(final_accuracy)
            eval_metrics[metric_key_prefix + "_loss"].append(final_loss)

            self.callback_manager.execute_callbacks("after_eval", final_accuracy, final_loss, self.global_step)

            if verbose:
                self.logger.info(f"Evaluation results at step [{self.global_step}] of {metric_key_prefix} dataset: "
                                 f"MLM Accuracy: {final_accuracy:.4f} --- MLM Loss: {final_loss:.4f}")

            return eval_metrics

    def train_epoch(self, epoch: int, training_metrics: dict, info_step: int = 100, num_steps: int = None,
                    verbose: bool = True, tqdm_iterator: tqdm = None):

        internal_steps = 0

        if self.config_change:
            warnings.warn("You changed the model config but did not reload the model. Your changes did therefore not "
                          "affect the model architecture.")

        break_flag = True if num_steps is None else False
        if num_steps is None:
            num_steps = float("inf")

        while internal_steps < num_steps:

            for batch_id, batch in enumerate(self.dataset):
                self.callback_manager.execute_callbacks("before_batch_processing", batch_id, batch)
                inputs, attention_mask = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)

                inputs, labels = self.mask_tokens(
                    inputs,
                    mlm_probability=self.optimization_config.mlm_probability,
                    mask_token_id=self.dataset.mask_token_id,
                    special_token_ids=self.dataset.special_token_ids,
                    n_tokens=self.dataset.tokenizer.vocab_size)

                mlm_output = self.model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                mlm_loss, mlm_logits = mlm_output.loss, mlm_output.logits
                training_metrics["loss"].append(mlm_loss.item())

                self.optimizer.zero_grad()
                mlm_loss.backward()

                if self.optimization_config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                mlm_grads = [p.grad.detach().flatten() for p in self.model.parameters() if p.grad is not None]
                mlm_grad_norm = torch.cat(mlm_grads).norm()

                self.optimizer.step()
                self.scheduler.step()

                with torch.no_grad():
                    masked_ids_mask = inputs == self.dataset.tokenizer.mask_token_id
                    predictions = mlm_logits.argmax(-1)
                    mlm_accuracy = (predictions[masked_ids_mask] == labels[masked_ids_mask]).float().mean()

                training_metrics["accuracy"].append(mlm_accuracy.item())
                training_metrics["gradient_norms"].append(mlm_grad_norm.item())
                current_lr = self.scheduler.get_last_lr()[0]
                training_metrics["learning_rates"].append(current_lr)

                self.callback_manager.execute_callbacks("on_batch_end", self.global_step, mlm_loss, mlm_accuracy.item(),
                                                        mlm_grad_norm.item(), current_lr)
                self.global_step += 1
                internal_steps += 1

                if tqdm_iterator is not None:
                    tqdm_iterator.update()

                if batch_id % info_step == 0 and verbose:
                    self.logger.info(
                        f"Results after {batch_id / self.iteration_steps_per_epoch:.4%} iterations of epoch {epoch + 1}:")
                    self.logger.info(f"MLM loss: {mlm_loss.item():.4f}")
                    self.logger.info(f"Gradient norm: {mlm_grad_norm:.4f}")
                    self.logger.info(f"Current learning rate: {current_lr}")
                    self.logger.info(f"Accuracy for masking task: {mlm_accuracy.item():.4f}")
                    self.logger.info("-" * 100)

                if internal_steps > num_steps:
                    break_flag = True
                    break

            if break_flag:
                break

        self.callback_manager.execute_callbacks("after_epoch", epoch, training_metrics)

        return training_metrics

    def train(self):
        self.logger.info("Starting with training...")
        self.training_state = []
        eval_step = 1000  # TODO: make variable --- currently not used --> later for evaluation
        info_step = 100  # TODO: make variable
        training_metrics = {
            "loss": [],
            "accuracy": [],
            "gradient_norms": [],
            "learning_rates": []
        }

        self.callback_manager.execute_callbacks("before_training")

        self.global_step = 0  # use global steps instead of epochs and batches
        for epoch in range(self.optimization_config.n_epochs):
            self.dataset.set_dataset_offsets(epoch)
            self.dataset.prepare_data_loader()
            self.callback_manager.execute_callbacks("on_epoch_start", epoch)

            self.train_epoch(epoch, training_metrics, info_step)

        self.logger.info("...training is finished, saving results and model.")

        save_path = self._create_directory_and_return_save_path(model_type="mlm")
        training_metrics_df = pd.DataFrame(training_metrics)
        training_metrics_df.to_csv(save_path + "training_metrics.csv", index=False)
        training_metrics_df.loc[:, ["loss"]].plot()
        plt.savefig(save_path + "loss.png")
        training_metrics_df.loc[:, ["accuracy"]].plot()
        plt.savefig(save_path + "accuracy.png")
        self.model.save_pretrained(save_path + "mlm_model")
        self.config.to_json(save_path + "model_config.json")

        self.logger.info("Results and model are saved.")




