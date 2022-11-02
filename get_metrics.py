import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="metrics.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.metrics_pipeline import get_metrics

    # Applies optional utilities
    utils.extras(config)

    # Evaluate model
    return get_metrics(config)


if __name__ == "__main__":
    main()
