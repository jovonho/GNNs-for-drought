from src.models.mlp import train_mlp
from src.data import DATAFOLDER_PATH


def main():
    train_mlp(root=DATAFOLDER_PATH, model_name="MLP")


if __name__ == "__main__":
    main()
