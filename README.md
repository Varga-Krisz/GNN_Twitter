# GNN_Twitter
Link prediction with graph neural networks on the Twitter dataset.

### Team Name: Me, myself, and I
### Members: Varga Krisztián - PVQIYU

## File descriptions
The `nagyHF.ipynb` notebook contains all the necessary code for this project,
including the data and requirement downloads, data preparation, and the model training.

The data is preprocessed into a DataFrame of bag-of-words in the `feature_bow.pkl` file extracted from `feature_bow.rar`,
and `node2vec_trained.pt` is a node embedding pretrained on the train data.

The required `Documentation.pdf` is currently missing due to time constraints and there being no results worth documenting.

## Related works and used resources
A non-exhustive list of resources used in creating this project:

- [Stack Overflow](https://stackoverflow.com/)
- [PyTorch](https://pytorch.org/) and the PyTorch forums
- [PyTorch Geometric](https://pyg.org/)
- [node2vec](https://snap.stanford.edu/node2vec/)
- A few Stanford University [Machine Learning with Graphs](https://web.stanford.edu/class/cs224w/) lectures

## Data
The Twitter dataset is available for download [here](https://snap.stanford.edu/data/ego-Twitter.html).

