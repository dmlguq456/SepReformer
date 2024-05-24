import argparse
import importlib

# Parse args
parser = argparse.ArgumentParser(
    description="Command to start PIT training, configured by .yaml files")
parser.add_argument(
    "--model",
    type=str,
    default="IQformer_v6_0_0",
    dest="model",
    help="Insert model name")
parser.add_argument(
    "--engine_mode",
    choices=["train", "test"],
    default="train",
    help="This option is used to chooose the mode")
args = parser.parse_args()

# Call target model
main_module = importlib.import_module(f"models.{args.model}.main")
main_module.main(args)