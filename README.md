# GNN_Twitter
Link prediction with graph neural networks on the Twitter dataset.

### Team Name: Me, myself, and I
### Members: Varga Krisztián - PVQIYU

## File descriptions
The `nagyHF.ipynb` notebook contains all the model training for this project, with all background functions and classes found in `model.py`.

The data is preprocessed into a pandas DataFrame by the `data_processing.ipynb` notebook.
The `feature_bow.pkl` file extracted from `feature_bow.rar` contains the bag-of-words vectors of the nodes,
and `node2vec_trained.pt` is a node embedding pretrained on the train data.

The `Documentation.pdf` gives a longer description of what this project is about.

## Related works and used resources
A non-exhustive list of resources used in creating this project:

- [Stack Overflow](https://stackoverflow.com/)
- [PyTorch](https://pytorch.org/) and the PyTorch forums
- [PyTorch Geometric](https://pyg.org/)
- [node2vec](https://snap.stanford.edu/node2vec/)
- A few Stanford University [Machine Learning with Graphs](https://web.stanford.edu/class/cs224w/) lectures

No LLMs have been used for development.

## Data
The Twitter dataset is available for download [here](https://snap.stanford.edu/data/ego-Twitter.html).

