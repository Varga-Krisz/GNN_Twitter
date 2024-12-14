# Imports
from IPython.display import HTML, display
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as gnn

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)


# -------------------------------------------------------------------------
# Train and test functions

# Display progress
def progress(value, max=100):
#	if value==max:
#		return ""
	return HTML(f"""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """)

def train(epoch, model, dataloader, optimizer, criterion, bar=None):
	model.train()
	
	# Reset progress
	bar.update(progress(0, len(dataloader)))

	
	epoch_loss = 0.0
	correct = 0
	total = 0

	for i, batch in enumerate(dataloader, 0):

		optimizer.zero_grad()
		outputs = model(batch.to(DEVICE))
		loss = criterion(outputs, batch.edge_label.to(DEVICE))
		loss.backward()
		optimizer.step()

		correct += (outputs.round()==batch.edge_label).sum().item()
		total += len(outputs)

		epoch_loss += loss.item()

		bar.update(progress(i+1, len(dataloader)))
	
	epoch_loss /= len(dataloader)
	epoch_acc = correct/total		
	return epoch_loss, epoch_acc


def eval(epoch, model, dataloader, criterion, bar=None):
	model.eval()

	# Reset progress
	bar.update(progress(0, len(dataloader)))

	epoch_loss = 0.0
	correct = 0
	total = 0

	for i, batch in enumerate(dataloader, 0):
		with torch.no_grad():
			outputs = model(batch.to(DEVICE))
			loss = criterion(outputs, batch.edge_label.to(DEVICE))
			correct += (outputs.round()==batch.edge_label).sum().item()
			total += len(outputs)

		epoch_loss += loss.item()

		bar.update(progress(i+1, len(dataloader)))

	
	epoch_loss /= len(dataloader)
	epoch_acc = correct/total
	return epoch_loss, epoch_acc


# -------------------------------------------------------------------------
# Embedding + dot product

class DotProduct(nn.Module):
    def __init__(self, num_features = 155522, embedding_dim = 20):
        super().__init__()
        self.emb = nn.EmbeddingBag(num_features,
                                   embedding_dim,
                                   max_norm=1,
                                   scale_grad_by_freq=True,
                                   mode="sum",
                                   padding_idx=0,
                                   )
    
    def forward(self, data):
        edge_index = data.edge_label_index
        user1_feat, user2_feat = data.BoW[edge_index]

        user1_vector = self.emb(user1_feat)
        user2_vector = self.emb(user2_feat)
        return (user1_vector * user2_vector).sum(1)+0.5
    

# -------------------------------------------------------------------------
# Embedding + 2 FC layers

class FCN(nn.Module):
	def __init__(self, num_features = 155522, embedding_dim = 20, hidden_dim = 32):
		super().__init__()
		self.emb = nn.EmbeddingBag(num_features, embedding_dim, max_norm=1, scale_grad_by_freq=True, padding_idx=0)

		self.lin1 = nn.Linear(2*embedding_dim, hidden_dim)
		#self.lin1 = nn.Linear(embedding_dim, hidden_dim)
		self.lin2 = nn.Linear(hidden_dim, 1)
		self.lin2.bias = nn.Parameter(torch.full(self.lin2.bias.shape, 0.5))
		self.nonlin = nn.ReLU()
		self.drop = nn.Dropout(0.5)

	def forward(self, data):
		edge_index = data.edge_label_index
		user1_feat, user2_feat = data.BoW[edge_index]

		user1_vector = self.emb(user1_feat)
		user2_vector = self.emb(user2_feat)

		x = torch.cat([user1_vector, user2_vector], dim=1)
		#x = user1_vector + user2_vector
		
		out = self.lin2(self.drop(self.nonlin(self.lin1(x)))).squeeze()
		return out


# -------------------------------------------------------------------------
# GCN

class GCN(torch.nn.Module):
	def __init__(self, num_features=155522, embedding_dim = 20, hidden_dim = 32, node2vec=None):
		super().__init__()
		self.emb = nn.EmbeddingBag(num_features,
                                   embedding_dim,
                                   max_norm=1,
                                   scale_grad_by_freq=True,
                                   mode="sum",
                                   padding_idx=0,
                                   )
		self.n2v = node2vec
		self.combine = nn.Linear(2*embedding_dim, self.n2v.embedding_dim)

		self.conv1 = gnn.GCNConv(self.n2v.embedding_dim, hidden_dim, flow="target_to_source")
		self.conv2 = gnn.GCNConv(hidden_dim, hidden_dim, flow="target_to_source")

		self.nonlin = nn.ReLU()
		self.drop = nn.Dropout(0.5)
		
		self.classifier = gnn.Linear(hidden_dim, 1)
		#self.classifier.bias = nn.Parameter(torch.full(self.classifier.bias.shape, 0.5))

	def forward(self, data):
		node_index, node_bow, edge_index, label_index = data.n_id, data.BoW, data.edge_index, data.edge_label_index
		
		# Combine node embeddings of the bag of words and node2vec
		x1 = self.emb(node_bow)
		x2 = self.n2v(node_index)
		x = torch.cat([x1,x2], dim=1)
		x = self.combine(self.nonlin(x))

		# cuda dies here...
		out = self.conv1(x, edge_index)
		out = self.conv2(self.drop(self.nonlin(out)), edge_index)

		out = self.classifier(self.drop(self.nonlin(out))).squeeze()
		out = out[label_index[0]]
		return out


# -------------------------------------------------------------------------
# Complete training function

def train_model(model, train_loader, val_loader, numEpoch = 5, optimizer="Adam", lr=0.01):
    model = model.to(DEVICE)
    optimizer = getattr(optim, optimizer)(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss().to(DEVICE)

    t_losses = []
    t_accs = []
    v_losses = []
    v_accs = []

    best_loss = 100.0
    best_acc = 0.0
    # Progress bar
    bar = display(progress(0, len(train_loader)), display_id=True)
    for epoch in range(numEpoch):
	
        # Train
        t_loss, t_acc = train(epoch, model, train_loader, optimizer, criterion, bar)
        t_losses.append(t_loss)
        t_accs.append(t_acc)
        print(f"Epoch {epoch+1}: Train loss: {t_loss:.4f}, - accuracy: {t_acc:.4f}")

        # Validate
        v_loss, v_acc = eval(epoch, model, val_loader, criterion, bar)
        v_losses.append(v_loss)
        v_accs.append(v_acc)
        print(F"Epoch {epoch+1}: Valid loss: {v_loss:.4f} - accuracy: {v_acc:.4f}")

        # LR scheduler
        # scheduler.step()

        # Compare to current best accuracy
        if v_loss <= best_loss:
            best_loss = v_loss
            best_acc = v_acc

    print(f"Best val_acc: {best_acc}")

    return t_losses, t_accs, v_losses, v_accs
