#!/usr/bin/env python3
import os
import uuid
import re
import logging


def rename_tensorboard_files(base_dir="ppo_tensorboard"):
    # Updated pattern to catch both UUID and hostname-based filenames
    pattern = re.compile(
        r"events\.out\.tfevents\.(\d+)\.([^.]+(?:\.[^.]+)*?)\.(\d+)\.(\d+)"
    )

    # Walk through all directories (including cryptocurrency subdirectories)
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                old_path = os.path.join(root, file)

                # Generate a UUID
                new_uuid = str(uuid.uuid4())

                # Get the match groups
                match = pattern.match(file)
                if match:
                    timestamp, _, pid, seq = match.groups()
                    # Create new filename using the matched groups
                    new_file = f"events.out.tfevents.{timestamp}.{new_uuid}.{pid}.{seq}"
                    new_path = os.path.join(root, new_file)

                    # Rename the file while preserving the cryptocurrency directory structure
                    try:
                        if old_path != new_path:  # Only rename if paths are different
                            if ".local." in old_path or not is_uuid(
                                old_path.split(".")[-3]
                            ):
                                os.rename(old_path, new_path)
                                print(f"Renamed: {old_path} → {new_path}")
                            else:
                                print(f"Skipping already UUID-formatted file: {file}")
                    except Exception as e:
                        print(f"Error renaming {old_path}: {e}")
                else:
                    print(f"Skipping {file} - doesn't match expected pattern")


def is_uuid(s):
    """Check if a string is a valid UUID"""
    try:
        uuid.UUID(str(s))
        return True
    except ValueError:
        return False


def organize_existing_files(base_dir="ppo_tensorboard"):
    """
    Organize existing TensorBoard files into cryptocurrency-specific directories
    based on the model files in the models directory
    """
    models_dir = "models"

    # Create a mapping of existing models to their cryptocurrencies
    crypto_models = {}
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith("_ppo.zip"):
                # Extract cryptocurrency symbol from model filename
                crypto_symbol = file.split("_")[0]
                crypto_models[crypto_symbol] = True

    # Ensure cryptocurrency directories exist
    for crypto in crypto_models:
        os.makedirs(os.path.join(base_dir, crypto), exist_ok=True)

    # Move existing files if they're not already in crypto-specific directories
    for root, dirs, files in os.walk(base_dir):
        if root == base_dir:  # Only process files in the base directory
            for file in files:
                if file.startswith("events.out.tfevents"):
                    # Try to determine which cryptocurrency this file belongs to
                    for crypto in crypto_models:
                        old_path = os.path.join(root, file)
                        new_dir = os.path.join(base_dir, crypto)
                        new_path = os.path.join(new_dir, file)

                        try:
                            if not os.path.exists(new_path):
                                os.rename(old_path, new_path)
                                print(f"Moved {file} to {crypto} directory")
                        except Exception as e:
                            print(f"Error moving {file}: {e}")


if __name__ == "__main__":
    organize_existing_files()  # First organize existing files
    rename_tensorboard_files()  # Then rename files while preserving structure
