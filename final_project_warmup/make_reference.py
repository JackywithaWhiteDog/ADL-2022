import sys
from pathlib import Path

SPLITS = ('dev', 'test', 'train')

if __name__ == '__main__':
    data_dir = Path(sys.argv[1])
    for split in SPLITS:
        split_dir = data_dir / split
        target_path = split_dir / "target.csv"
        data = []
        with open(target_path, "r") as f:
            for line in f:
                data.append(line[line.find(",")+1:].strip())
        ref_path = split_dir / "reference.txt"
        with open(ref_path, "w") as f:
            if len(data) > 0:
                f.write('\n'.join(data))
