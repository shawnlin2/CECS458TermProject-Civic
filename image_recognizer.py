from datasets import load_dataset

ds = load_dataset("Programmer-RD-AI/road-issues-detection-dataset")
example = ds['train'][6312]
print(example)
img = example['image']
img.show()