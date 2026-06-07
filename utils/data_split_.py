import torch
import os
import random

LABELS_PATH = './datasets/pt_file'

def split_dataset(fold_num, split, pt_path=LABELS_PATH, sub_val_ratio=0.25, sub_val=False, seed=42):
    """
    Dataset splitting function: reads the corresponding .pt file based on fold number and split type, and filters out invalid IDs.
    When sub_val=True, the original val set is no longer used; instead, a new train/val split is created from the original train set.

    Args:
        fold_num (int): Fold number, e.g., 1, 2, 3...
        split (str): Split type, only supports 'train' / 'val' / 'test' / 'external_test'
        pt_path (str): Path to the folder containing .pt files
        sub_val_ratio (float): Ratio of samples taken from the training set for validation, default is 0.25
        sub_val (bool): Whether to re-split the training set into train/val, default is False
        seed (int): Random seed, default is 42

    Returns:
        list: List of valid IDs after filtering
    """
    # 1. Validate input parameters
    valid_splits = ['train', 'val', 'test', 'external_test']
    if split not in valid_splits:
        raise ValueError(f"split must be one of {valid_splits}")

    if not isinstance(fold_num, int) or fold_num < 1:
        raise ValueError("fold_num must be an integer >= 1")

    if not (0 < sub_val_ratio < 1):
        raise ValueError("sub_val_ratio must be in (0, 1)")

    try:
        # 2. Load the full list of valid IDs
        all_valid_ids = torch.load(
            os.path.join(pt_path, "all_patient_labels.pt"),
            weights_only=False
        )
        all_valid_ids = all_valid_ids.tolist() if hasattr(all_valid_ids, 'tolist') else list(all_valid_ids)
        all_valid_ids = set(all_valid_ids)

        # 3. Determine which file to load based on sub_val and split
        # Split the original train set into: new train + sub-validation set
        if sub_val and split in ["train", "val"]:
            # When sub_val=True, both train and val are re-split from the original train set
            data_file = os.path.join(pt_path, f"train_fold{fold_num}.pt")
            raw_ids = list(torch.load(data_file, weights_only=False).keys())

            # Filter valid IDs
            filtered_ids = [pid for pid in raw_ids if pid in all_valid_ids]

            # To ensure consistent splits across train/val calls:
            # 1) Sort to avoid instability from key order
            # 2) Use a fixed seed combined with fold_num to create a local random generator
            filtered_ids = sorted(set(filtered_ids))
            rng = random.Random(seed + fold_num)
            rng.shuffle(filtered_ids)

            val_size = max(1, int(len(filtered_ids) * sub_val_ratio))
            new_val_ids = filtered_ids[:val_size]
            new_train_ids = filtered_ids[val_size:]

            # Edge case protection: prevent empty new_train_ids
            if len(new_train_ids) == 0:
                raise ValueError(
                    f"Training set is empty after splitting. Please reduce sub_val_ratio. "
                    f"Current sample count={len(filtered_ids)}, sub_val_ratio={sub_val_ratio}"
                )

            if split == "train":
                print(
                    f"✅ sub_val=True | Fold{fold_num} original_train={len(filtered_ids)} "
                    f"-> new_train={len(new_train_ids)}, new_val={len(new_val_ids)}"
                )
                return new_train_ids
            else:  # split == "val"
                # print(
                #     f"✅ sub_val=True | Fold{fold_num} original_train={len(filtered_ids)} "
                #     # f"-> new_train={len(new_train_ids)}, new_val={len(new_val_ids)}"
                # )
                return new_val_ids

        else:
            # 4. Original logic: directly load the corresponding split file
            if split == "test":
                data_file = os.path.join(pt_path, f"external_test_fold{fold_num}.pt")
            else:
                data_file = os.path.join(pt_path, f"internal_test_fold{fold_num}.pt")

            raw_ids = list(torch.load(data_file, weights_only=False).keys())

            # Filter valid IDs
            filtered_ids = [pid for pid in raw_ids if pid in all_valid_ids]
            filtered_ids = sorted(set(filtered_ids))

            # print(f"✅ Using original {split} set | Fold{fold_num} {split} count: {len(filtered_ids)}")
            return filtered_ids

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e.filename}, please check the file path")
    except Exception as e:
        raise RuntimeError(f"Dataset splitting failed: {str(e)}")


if __name__ == "__main__":
    # Configuration parameters
    LABELS_PATH = './datasets/pt_file'
    fold_num = 1
    train_ids = split_dataset(fold_num=fold_num, split="train", pt_path=LABELS_PATH, sub_val_ratio=0.25, sub_val=True)
    val_ids = split_dataset(fold_num=fold_num, split="val", pt_path=LABELS_PATH, sub_val_ratio=0.25, sub_val=True)
    test_ids = split_dataset(fold_num=fold_num, split="val", pt_path=LABELS_PATH, sub_val=False)
    external_test_ids = split_dataset(fold_num=fold_num, split="test", pt_path=LABELS_PATH, sub_val=False)

    print(f"Training set: {len(train_ids)}")
    print(f"Validation set: {len(val_ids)}")
    print(f"Test set: {len(test_ids)}")
    print(f"External test set: {len(external_test_ids)}")
    print(f"Total samples: {len(train_ids) + len(val_ids) + len(test_ids) + len(external_test_ids)}")