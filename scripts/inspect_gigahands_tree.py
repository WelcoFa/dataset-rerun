from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_ROOT = REPO_ROOT / "data" / "gigahands"


def tree(dir_path, prefix=""):
    contents = list(dir_path.iterdir())
    pointers = ["|- "] * (len(contents) - 1) + ["`- "]

    for pointer, path in zip(pointers, contents):
        print(prefix + pointer + path.name)

        if path.is_dir():
            extension = "|  " if pointer == "|- " else "   "
            tree(path, prefix + extension)


folder = DATA_ROOT / "gigahands_demo_all"
print(folder)
tree(folder)
