import tensorflow as tf
from enhance.dataset import Dataset
from enhance.optimiser import Optimiser
import argparse
import os
import wandb

def main():
    parser = argparse.ArgumentParser(description='Setup variables')

    parser.add_argument("data_dir", type=str,
                        help='Expects a data dir with subfolders train/, val/ with containing images')

    parser.add_argument("checkpoint_dir", type=str,
                        help='Directory to log model checkpoints')

    parser.add_argument("--buffer_size", type=int, default=400,
                        help='Directory to log model checkpoints')

    parser.add_argument("--batch_size", type=int, default=1,
                        help='Directory to log model checkpoints')

    parser.add_argument("--patches_per_image", type=int, default=10,
                        help='Directory to log model checkpoints')

    args = parser.parse_args()

    config = {}
    for key in args.__dict__:
        config[key] = args.__dict__[key]


    """
    Init WandB
    """
    wandb.init(project="stargazer-enhance", config=config)
    wandb.define_metric(name="train/epoch")
    wandb.define_metric(name="train/*",
                        step_metric="train/epoch")

    """
    Load the datasets
    """
    dataset_train = Dataset(os.path.join(args.data_dir, "train"))
    dataset_val = Dataset(os.path.join(args.data_dir, "val"))

    data_train = dataset_train.get_dataset(patches_per_image=args.patches_per_image)
    data_val = dataset_val.get_dataset(patches_per_image=args.patches_per_image)

    dataset_train = tf.data.Dataset.from_tensor_slices(data_train)
    dataset_train = dataset_train.shuffle(args.buffer_size)
    dataset_train = dataset_train.batch(args.batch_size)

    dataset_val = tf.data.Dataset.from_tensor_slices(data_val)
    dataset_val = dataset_val.shuffle(args.buffer_size)
    dataset_val = dataset_val.batch(args.batch_size)


    """
    Define optimiser
    """
    optimiser = Optimiser(args.checkpoint_dir)
    
    optimiser.train(dataset_train, dataset_val, steps=500)


if __name__ == "__main__":
    main()
