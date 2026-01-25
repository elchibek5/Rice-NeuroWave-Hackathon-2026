import argparse
import os
import shutil

# merges multiple training folders into one output folder, without overwriting duplicates
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_train_dir", required=True, help="Output Training directory (will be created)")
    ap.add_argument("--train_dirs", nargs="+", required=True, help="One or more Training directories to merge")
    args = ap.parse_args()

    os.makedirs(args.out_train_dir, exist_ok=True)

    copied = 0
    for d in args.train_dirs:
        for f in os.listdir(d):
            src = os.path.join(d, f)
            dst = os.path.join(args.out_train_dir, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied += 1

    print(f"Copied {copied} files into {args.out_train_dir}")
    # Basic sanity counts
    x = len([f for f in os.listdir(args.out_train_dir) if f.startswith("X_train_")])
    y = len([f for f in os.listdir(args.out_train_dir) if f.startswith("y_train_")])
    print(f"Counts: X_train={x}, y_train={y}")

if __name__ == "__main__":
    main()
