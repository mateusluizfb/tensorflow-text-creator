from textgenrnn import textgenrnn
from pathlib import Path

if Path("./textgenrnn_weights.hdf5").is_file():
    textgen = textgenrnn()
else:
    textgen = textgenrnn('textgenrnn_weights.hdf5')
    textgen.train_from_file('data.txt', num_epochs=2)

textgen.generate(1, temperature=0.5, return_as_list=True)

print(text)
