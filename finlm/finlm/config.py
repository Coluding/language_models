from dataclasses import dataclass, asdict
from typing import Any, Dict
import yaml
import json

# The DatasetConfig, ModelConfig and OptimizationConfig are united in the FinLMConfig which is used for pretraining
# The FinetuningConfig is used for downstreaming tasks



@dataclass
class DatasetConfig:

    """
    Configuration class for the dataset used in the FinLM model.

    Attributes
    ----------
    tokenizer_path : str
        Path to the tokenizer to be used for text preprocessing.
    max_sequence_length : int
        Maximum length of input sequences after tokenization.
    db_name : str
        Name of the database where the dataset is stored.
    batch_size : int
        Number of sequences to include in each batch.
    database_retrieval : dict[str, dict[str, int]]
        Dictionary specifying retrieval parameters from the database, including limits and offsets.

    Methods
    -------
    from_dict(data: Dict[str, Any]) -> 'DatasetConfig'
        Creates an instance of DatasetConfig from a dictionary.
    """

    tokenizer_path: str 
    max_sequence_length: int 
    db_name: str 
    batch_size: int 
    database_retrieval: dict[str, dict[str, int]]



@dataclass
class ModelConfig:

    """
    Configuration class for the FinLM model architecture.

    Attributes
    ----------
    embedding_size : int, optional
        Size of the embedding layer (default is 128).
    hidden_size : int, optional
        Size of the hidden layers (default is 256).
    num_hidden_layers : int, optional
        Number of hidden layers in the model (default is 12).
    num_attention_heads : int, optional
        Number of attention heads in each attention layer (default is 4).
    intermediate_size : int, optional
        Size of the intermediate layer in the transformer (default is 1024).
    max_position_embeddings: int, optional
        The maximum sequence length that this model might ever be used with.
    generator_size : float, optional
        Scaling factor for the generator size (default is 0.25).
    generator_layer_size : float, optional
        Scaling factor for the generator layer size (default is 1.0).
    model_name : str, optional
        Name of the model architecture (default is "electra").
    attention_window: int, optional
        Size of the attention window (default is 256). Only used for Longformer.

    Methods
    -------
    from_dict(data: Dict[str, Any]) -> 'ModelConfig'
        Creates an instance of ModelConfig from a dictionary.
    """

    embedding_size: int = 128
    hidden_size: int = 256
    num_hidden_layers: int = 12
    num_attention_heads: int = 4
    intermediate_size: int = 1024
    max_position_embeddings: int = 256
    generator_size: float = 0.25
    generator_layer_size: float = 1.0
    model_name: str = "electra"
    attention_window: int = 128


@dataclass
class EncoderConfig:
    model_name: str
    embedding_size: int = 128
    hidden_size: int = 256
    num_hidden_layers: int = 12
    num_attention_heads: int = 4
    intermediate_size: int = 1024
    max_position_embeddings: int = 256
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1


@dataclass
class OptimizationConfig:

    """
    Configuration class for the optimization settings of the FinLM model.

    Attributes
    ----------
    learning_rate : float, optional
        Learning rate for the optimizer (default is 0.0001).
    n_epochs : int, optional
        Number of training epochs (default is 1).
    lr_scheduler_warm_up_steps : int, optional
        Number of warm-up steps for the learning rate scheduler (default is 1000).
    mlm_probability : float, optional
        Probability of masking tokens for masked language modeling (default is 0.15).
    use_gradient_clipping : bool, optional
        Whether to use gradient clipping (default is True).
    discriminator_weight : int, optional
        Weight for the discriminator loss (default is 50).
    discriminator_sampling : str, optional
        Sampling strategy for the discriminator (default is "multinomial").

    Methods
    -------
    from_dict(data: Dict[str, Any]) -> 'OptimizationConfig'
        Creates an instance of OptimizationConfig from a dictionary.
    """

    learning_rate: float = 0.0001
    n_epochs: int = 1
    lr_scheduler_warm_up_steps: int = 1000
    mlm_probability: float = 0.15
    use_gradient_clipping: bool = True
    discriminator_weight: int = 50
    discriminator_sampling: str = "multinomial"

@dataclass
class FinLMConfig:

    """
    Configuration class for the FinLM model, combining dataset, model, and optimization configurations.

    Attributes
    ----------
    dataset_config : DatasetConfig
        Configuration for the dataset.
    model_config : ModelConfig
        Configuration for the model architecture.
    optimization_config : OptimizationConfig
        Configuration for the optimization settings.
    save_models_and_results_to : str
        Path where models and results should be saved.

    Methods
    -------
    from_yaml(config_file_path: str, save_root_path: str) -> 'FinLMConfig'
        Loads the configuration from a YAML file.
    to_dict() -> Dict[str, Any]
        Converts the configuration to a dictionary.
    to_json(file_path: str) -> None
        Saves the configuration as a JSON file.
    """

    dataset_config: DatasetConfig
    model_config: ModelConfig
    optimization_config: OptimizationConfig
    save_models_and_results_to: str

    @classmethod
    def from_yaml(cls, config_file_path: str, save_root_path: str) -> 'FinLMConfig':

        """
        Loads the configuration from a YAML file.

        Parameters
        ----------
        config_file_path : str
            The path to the YAML file containing the configuration.
        save_root_path : str
            The root path where models and results will be saved.

        Returns
        -------
        FinLMConfig
            An instance of FinLMConfig initialized with the data from the YAML file.
        """

        with open(config_file_path, 'r') as file:
            config_data = yaml.safe_load(file)

        dataset_config = DatasetConfig(**config_data.get('dataset_config', {}))
        model_config = ModelConfig(**config_data.get('model_config', {}))
        optimization_config = OptimizationConfig(**config_data.get('optimization_config', {}))

        return cls(
            dataset_config=dataset_config,
            model_config=model_config,
            optimization_config=optimization_config,
            save_models_and_results_to = save_root_path
        )
    
    def to_dict(self) -> Dict[str, Any]:

        """
        Converts the configuration to a dictionary.

        Returns
        -------
        Dict[str, Any]
            A dictionary representation of the FinLMConfig.
        """

        return {
            'dataset_config': asdict(self.dataset_config),
            'model_config': asdict(self.model_config),
            'optimization_config': asdict(self.optimization_config)
        }

    def to_json(self, file_path: str) -> None:

        """
        Saves the configuration as a JSON file.

        Parameters
        ----------
        file_path : str
            The path where the JSON file will be saved.
        """

        with open(file_path, 'w') as file:
            json.dump(self.to_dict(), file, indent=4)


@dataclass
class FintuningConfig:
    model_path: str
    num_labels: int
    tokenizer_path: str
    max_sequence_length: int
    text_column: str
    dataset_name: str
    dataset_columns: list[str]
    training_data_fraction: float
    batch_size: int
    max_sequences: int
    n_splits: int
    early_stopping_patience: int
    n_epochs: Dict[str, Any]
    learning_rate: Dict[str, Any]
    classifier_dropout: Dict[str, Any]
    warmup_step_fraction: Dict[str, Any]
    use_gradient_clipping: Dict[str, Any]
    save_path: str


    @classmethod
    def from_yaml(cls, config_file_path: str):
        with open(config_file_path, 'r') as yaml_file:
            data = yaml.safe_load(yaml_file)

        return cls(**data)
    
    def to_dict(self):
        return asdict(self)
    
    def to_json(self, file_path: str) -> None:

        """
        Saves the configuration as a JSON file.

        Parameters
        ----------
        file_path : str
            The path where the JSON file will be saved.
        """

        with open(file_path, 'w') as file:
            json.dump(self.to_dict(), file, indent=4)

