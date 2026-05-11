import json
import random

random.seed(42)

# 240 unikatnih števil (200 train + 40 val)
all_numbers = random.sample(range(1, 701), 240)

train_numbers = all_numbers[:200]
val_numbers = all_numbers[200:]

train_ids = [str(num) for num in train_numbers]
val_ids = [str(num) for num in val_numbers]

# struktura za splits_final_200.json
splits = [
    {
        "train": train_ids,
        "val": val_ids
    }
]

# shrani v datoteko
with open("workdir/splits_final_200.json", "w") as f:
    json.dump(splits, f, indent=4)

print("splits_final_200.json uspešno ustvarjen!")