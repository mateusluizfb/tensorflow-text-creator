from textgenrnn import textgenrnn
from pathlib import Path

try:
    Path("./textgenrnn_weights.hdf5")
    textgen = textgenrnn('textgenrnn_weights.hdf5')
except IOError:
    textgen = textgenrnn()
    textgen.train_from_file('data.txt', num_epochs=2)

text = textgen.generate(1, temperature=0.5, return_as_list=True)

print(text)
