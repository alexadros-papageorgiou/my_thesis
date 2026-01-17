import numpy as np
import pandas as pd

attack_path = "/kaggle/input/cic-ddos2019-30gb-full-dataset-csv-files/01-12/DrDoS_DNS.csv"
benign_path = "/kaggle/input/cic-ddos2019-30gb-full-dataset-csv-files/03-11/UDP.csv"

df_attack = pd.read_csv(attack_path).sample(30000, random_state=42)
df_benign = pd.read_csv(benign_path).sample(30000, random_state=42)

df_attack["label"] = 1
df_benign["label"] = 0

df = pd.concat([df_attack, df_benign], ignore_index=True).dropna()

import numpy as np

id_cols = [
    "Flow ID", " Timestamp", " Label", "Unnamed: 0"
]

# Προσοχή: οι στήλες έχουν κενά μπροστά
entity_cols = [
    " Source IP", " Destination IP",
    " Source Port", " Destination Port",
    " Protocol"
]


df = df.drop(columns=[c for c in id_cols if c in df.columns], errors="ignore")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=entity_cols + ["label"], inplace=True)

from sklearn.preprocessing import StandardScaler

num_cols = df.select_dtypes(include=["number"]).columns.tolist()

# Βγάζουμε entity columns και το label από τα numeric features
num_cols = [c for c in num_cols if c not in ["label"]]
for c in [" Source Port", " Destination Port", " Protocol"]:
    if c in num_cols:
        num_cols.remove(c)
# Καθαρισμός NaN στα numeric flow features
df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
df[num_cols] = df[num_cols].fillna(0.0)

scaler = StandardScaler()
X_flow_scaled = scaler.fit_transform(df[num_cols].values)

X_flow = X_flow_scaled.astype(np.float32)
y_flow = df["label"].astype(np.int64).values

np.isnan(X_flow).sum(), np.isinf(X_flow).sum()

def make_id_map(series: pd.Series):
    uniq = series.astype(str).unique()
    return {v: i for i, v in enumerate(uniq)}

src_ip = df[" Source IP"].astype(str)
dst_ip = df[" Destination IP"].astype(str)
all_ip = pd.concat([src_ip, dst_ip], ignore_index=True)
ip_map = make_id_map(all_ip)

src_port = df[" Source Port"].astype(str)
dst_port = df[" Destination Port"].astype(str)
all_port = pd.concat([src_port, dst_port], ignore_index=True)
port_map = make_id_map(all_port)

proto = df[" Protocol"].astype(str)
proto_map = make_id_map(proto)

flow_ids = np.arange(len(df), dtype=np.int64)

src_ip_id = src_ip.map(ip_map).astype(np.int64).values
dst_ip_id = dst_ip.map(ip_map).astype(np.int64).values

src_port_id = src_port.map(port_map).astype(np.int64).values
dst_port_id = dst_port.map(port_map).astype(np.int64).values

proto_id = proto.map(proto_map).astype(np.int64).values

import torch
from torch_geometric.data import HeteroData

data = HeteroData()

# Node features
data["flow"].x = torch.tensor(X_flow, dtype=torch.float32)
data["flow"].y = torch.tensor(y_flow, dtype=torch.long)

# Για ip/port/proto δεν βάζουμε features (θα γίνουν embeddings από το μοντέλο)
data["ip"].num_nodes = len(ip_map)
data["port"].num_nodes = len(port_map)
data["proto"].num_nodes = len(proto_map)

# Edges: flow -> ip (src, dst)
data["flow", "src_ip", "ip"].edge_index = torch.tensor(
    np.vstack([flow_ids, src_ip_id]), dtype=torch.long
)
data["flow", "dst_ip", "ip"].edge_index = torch.tensor(
    np.vstack([flow_ids, dst_ip_id]), dtype=torch.long
)

# Edges: flow -> port (src, dst)
data["flow", "src_port", "port"].edge_index = torch.tensor(
    np.vstack([flow_ids, src_port_id]), dtype=torch.long
)
data["flow", "dst_port", "port"].edge_index = torch.tensor(
    np.vstack([flow_ids, dst_port_id]), dtype=torch.long
)

# Edges: flow -> proto
data["flow", "uses_proto", "proto"].edge_index = torch.tensor(
    np.vstack([flow_ids, proto_id]), dtype=torch.long
)

data["ip", "rev_src_ip", "flow"].edge_index = torch.tensor(
    np.vstack([src_ip_id, flow_ids]), dtype=torch.long
)

data["ip", "rev_dst_ip", "flow"].edge_index = torch.tensor(
    np.vstack([dst_ip_id, flow_ids]), dtype=torch.long
)

data["port", "rev_src_port", "flow"].edge_index = torch.tensor(
    np.vstack([src_port_id, flow_ids]), dtype=torch.long
)

data["port", "rev_dst_port", "flow"].edge_index = torch.tensor(
    np.vstack([dst_port_id, flow_ids]), dtype=torch.long
)

data["proto", "rev_uses_proto", "flow"].edge_index = torch.tensor(
    np.vstack([proto_id, flow_ids]), dtype=torch.long
)

from sklearn.model_selection import train_test_split

idx = np.arange(data["flow"].num_nodes)
idx_train, idx_test = train_test_split(
    idx,
    test_size=0.3,
    random_state=42,
    stratify=y_flow
)

train_mask = torch.zeros(data["flow"].num_nodes, dtype=torch.bool)
test_mask = torch.zeros(data["flow"].num_nodes, dtype=torch.bool)

train_mask[idx_train] = True
test_mask[idx_test] = True

data["flow"].train_mask = train_mask
data["flow"].test_mask = test_mask

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HGTConv, Linear

# -------------- HGT --------------
class SimpleHGT(nn.Module):
    def __init__(self, metadata, hidden=64, out_classes=2):
        super().__init__()

        self.flow_lin = Linear(-1, hidden)

        self.ip_emb = nn.Embedding(data["ip"].num_nodes, hidden)
        self.port_emb = nn.Embedding(data["port"].num_nodes, hidden)
        self.proto_emb = nn.Embedding(data["proto"].num_nodes, hidden)

        self.conv = HGTConv(hidden, hidden, metadata, heads=2)
        self.cls = nn.Linear(hidden, out_classes)

    def forward(self, data):
        x_dict = {
            "flow": self.flow_lin(data["flow"].x),
            "ip": self.ip_emb.weight,
            "port": self.port_emb.weight,
            "proto": self.proto_emb.weight,
        }

        x_dict = self.conv(x_dict, data.edge_index_dict)
        x_flow = F.relu(x_dict["flow"])

        return self.cls(x_flow)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)

model = SimpleHGT(data.metadata()).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ------------------- Evaluation -------------------
def evaluate():
    model.eval()
    with torch.no_grad():
        logits = model(data)
        pred = logits.argmax(dim=1)

        y_true = data["flow"].y
        mask = data["flow"].test_mask

        correct = (pred[mask] == y_true[mask]).sum().item()
        total = mask.sum().item()
        return correct / total

# --------------- Γίνεται η εκπαίδευση ---------------
for epoch in range(1, 11):
    model.train()
    opt.zero_grad()

    logits = model(data)
    loss = F.cross_entropy(
        logits[data["flow"].train_mask],
        data["flow"].y[data["flow"].train_mask]
    )
    loss.backward()
    opt.step()

    acc = evaluate()
    print(f"Epoch {epoch:02d} | loss={loss.item():.4f} | test_acc={acc:.4f}")


from sklearn.metrics import classification_report, confusion_matrix

model.eval()
with torch.no_grad():
    logits = model(data)
    pred = logits.argmax(dim=1).cpu().numpy()

y_true = data["flow"].y.cpu().numpy()
mask = data["flow"].test_mask.cpu().numpy()

print(classification_report(y_true[mask], pred[mask]))
confusion_matrix(y_true[mask], pred[mask])
