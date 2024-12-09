#!/usr/bin/env python3
import os
import uuid
import re


def rename_tensorboard_files(base_dir="ppo_tensorboard"):
    # Pattern that matches either format:
    # 1. UUID format: events.out.tfevents.1731781541.uuid.4812.8
    # 2. Hostname format: events.out.tfevents.1733724779.samm-mbp.local.38418.8
    pattern = re.compile(
        r"events\.out\.tfevents\.(\d+)\.([^.]+(?:\.[^.]+)*?)\.(\d+)\.(\d+)"
    )

    # Walk through all directories
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

                    # Rename the file
                    try:
                        if old_path != new_path:  # Only rename if the path would change
                            os.rename(old_path, new_path)
                            print(f"Renamed: {old_path} → {new_path}")
                    except Exception as e:
                        print(f"Error renaming {old_path}: {e}")
                else:
                    print(f"Skipping {file} - doesn't match expected pattern")


if __name__ == "__main__":
    rename_tensorboard_files()
