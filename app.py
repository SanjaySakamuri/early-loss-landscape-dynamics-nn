import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Loss Landscape Dynamics", layout="wide", page_icon="🏔️")

st.title("Neural Network Loss Landscapes: A Comparative Study of SGD, RMSProp, and Adam")
st.markdown("""
**Parameter Space vs. Feature Space Dynamics.** Watch how different optimization algorithms navigate the high-dimensional empirical risk surface (projected via Principal Directions) and how their physical movement alters the model's real-world decision boundary.
""")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MicroNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

@st.cache_data
def get_real_data(noise=0.15):
    data = load_breast_cancer()
    
    X = data.data
    y = data.target

    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).view(-1, 1).to(device)
    return X, y, X_tensor, y_tensor

def get_weights(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()

def set_weights(model, vector):
    torch.nn.utils.vector_to_parameters(vector, model.parameters())

def calculate_loss(model, X_tensor, y_tensor, criterion):
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
    return loss.item()

def train_model(model, X_tensor, y_tensor, lr, epochs, optimizer_name):
    criterion = nn.BCELoss()
    
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "RMSProp":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    trajectory, losses, grad_norms = [], [], []
    model.train()
    trajectory.append(get_weights(model))
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_tensor)
        loss = criterion(out, y_tensor)
        loss.backward()
        
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in model.parameters() if p.grad is not None]))
        grad_norms.append(total_norm.item())
        losses.append(loss.item())
        
        optimizer.step()
        trajectory.append(get_weights(model))
            
    return trajectory, losses, grad_norms

def compute_landscape(model, base_weights, d1, d2, X_tensor, y_tensor, extent, grid_size):
    alphas = np.linspace(-extent, extent, grid_size)
    betas = np.linspace(-extent, extent, grid_size)
    X_grid, Y_grid = np.meshgrid(alphas, betas)
    Z = np.zeros_like(X_grid)
    
    criterion = nn.BCELoss()
    
    for i in range(grid_size):
        for j in range(grid_size):
            new_weights = base_weights + alphas[i] * d1 + betas[j] * d2
            set_weights(model, new_weights)
            Z[i, j] = calculate_loss(model, X_tensor, y_tensor, criterion)
            
    set_weights(model, base_weights)
    return alphas, betas, Z

def project_trajectory(trajectory, base_weights, d1, d2):
    x_coords, y_coords = [], []
    for w in trajectory:
        diff = w - base_weights
        x_coords.append(torch.dot(diff, d1).item())
        y_coords.append(torch.dot(diff, d2).item())
    return x_coords, y_coords

with st.sidebar:
    st.header("Hyperparameters")
    
    optimizer_choice = st.radio(
        "Optimizer Algorithm", 
        ["SGD", "RMSProp", "Adam"], 
        index=2,
        horizontal=True
    )
    
    st.markdown("### 📖 Algorithm Mechanics")
    if optimizer_choice == "SGD":
        st.info("**Stochastic Gradient Descent (SGD):** The classic workhorse. It takes rigid, uniform steps down the steepest slope of the current batch. Without momentum, it struggles in ravines and often produces noisy, zig-zag trajectories, making it slower to converge in complex landscapes.")
    elif optimizer_choice == "RMSProp":
        st.warning("**Root Mean Square Propagation (RMSProp):** Designed to tackle diminishing learning rates. It divides the gradient by a running average of its recent magnitude. Watch how it adaptively scales its steps to plow straight through flat plateaus and narrow valleys without exploding.")
    elif optimizer_choice == "Adam":
        st.success("**Adaptive Moment Estimation (Adam):** The industry standard. It elegantly combines RMSProp's adaptive scaling with a physics-like *momentum* term (tracking past gradients). Notice how it smoothly carves curved paths through the landscape and accelerates effortlessly towards the minima.")

    st.markdown("---")
    learning_rate = st.select_slider("Learning Rate", options=[0.01, 0.05, 0.1, 0.5, 1.0], value=0.1)
    epochs = st.slider("Epochs (Full Batch)", min_value=10, max_value=300, value=150, step=10)
    data_noise = st.slider("Dataset Complexity (Noise)", 0.0, 0.5, 0.15)
    
    st.markdown("---")
    st.markdown("**Performance Tip:** Because we use a Micro-MLP and full-batch training on 400 points, the math computes almost instantly.")

X_np, y_np, X_tensor, y_tensor = get_synthetic_data(data_noise)

model = MicroNet().to(device)
init_weights = get_weights(model)

traj, losses, grads = train_model(model, X_tensor, y_tensor, learning_rate, epochs, optimizer_choice)
opt_weights = get_weights(model)

d1 = opt_weights - init_weights
dist = torch.norm(d1).item()
if dist < 1e-5: d1 = torch.randn_like(opt_weights) 
d1 = d1 / torch.norm(d1)

d2 = torch.randn_like(opt_weights)
d2 = d2 - torch.dot(d2, d1) * d1
d2 = d2 / torch.norm(d2)

extent = max(dist * 1.5, 2.0)
grid_resolution = 20

alphas, betas, Z = compute_landscape(model, opt_weights, d1, d2, X_tensor, y_tensor, extent, grid_resolution)
x_proj, y_proj = project_trajectory(traj, opt_weights, d1, d2)

z_proj = []
for w in traj:
    set_weights(model, w)
    z_proj.append(calculate_loss(model, X_tensor, y_tensor, nn.BCELoss()))
set_weights(model, opt_weights)

xx, yy = np.meshgrid(np.linspace(X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5, 50),
                     np.linspace(X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5, 50))
grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)

model.eval()
with torch.no_grad():
    Z_boundary = model(grid_tensor).cpu().numpy().reshape(xx.shape)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 1. 3D Parameter Surface")
    fig_3d = go.Figure(data=[go.Surface(z=Z, x=alphas, y=betas, colorscale='Viridis', opacity=0.8, showscale=False)])
    fig_3d.add_trace(go.Scatter3d(
        x=x_proj, y=y_proj, z=z_proj, mode='lines+markers',
        marker=dict(size=3, color='red'), line=dict(color='red', width=4), name='Trajectory'
    ))
    fig_3d.update_layout(height=400, margin=dict(l=0, r=0, b=0, t=0), scene=dict(
        xaxis_title='d1 (Start→End)', yaxis_title='d2 (Orthogonal)', zaxis_title='Loss'
    ))
    st.plotly_chart(fig_3d, width="stretch")

with col2:
    st.markdown("### 2. Optimization Trajectory (Contour)")
    fig_contour = go.Figure(data=[go.Contour(z=Z, x=alphas, y=betas, colorscale='Viridis', showscale=False)])
    fig_contour.add_trace(go.Scatter(x=x_proj, y=y_proj, mode='lines+markers', marker=dict(size=4, color='red'), line=dict(color='red', width=2), name='Path'))
    fig_contour.add_trace(go.Scatter(x=[-dist], y=[0], mode='markers', marker=dict(size=12, symbol='star', color='yellow'), name='Start'))
    fig_contour.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(size=10, symbol='x', color='black'), name='Optima'))
    fig_contour.update_layout(height=400, margin=dict(l=0, r=0, b=0, t=0), xaxis_title="Principal Direction 1", yaxis_title="Orthogonal Direction 2")
    st.plotly_chart(fig_contour, width="stretch")

st.divider()

col3, col4 = st.columns(2)

with col3:
    st.markdown("### 3. Final Decision Boundary (Feature Space)")
    fig_db = go.Figure(data=[go.Contour(x=xx[0,:], y=yy[:,0], z=Z_boundary, colorscale='RdBu', opacity=0.4, showscale=False)])
    fig_db.add_trace(go.Scatter(x=X_np[y_np==0, 0], y=X_np[y_np==0, 1], mode='markers', marker=dict(color='red', line=dict(color='white', width=1)), name='Class 0'))
    fig_db.add_trace(go.Scatter(x=X_np[y_np==1, 0], y=X_np[y_np==1, 1], mode='markers', marker=dict(color='blue', line=dict(color='white', width=1)), name='Class 1'))
    fig_db.update_layout(height=400, margin=dict(l=0, r=0, b=0, t=0), xaxis_title="Feature X1", yaxis_title="Feature X2")
    st.plotly_chart(fig_db, width="stretch")

with col4:
    st.markdown("### 4. Convergence Diagnostics")
    fig_metrics = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    fig_metrics.add_trace(go.Scatter(y=losses, mode='lines', name='Loss', line=dict(color='purple')), row=1, col=1)
    fig_metrics.add_trace(go.Scatter(y=grads, mode='lines', name='Grad Norm', line=dict(color='orange')), row=2, col=1)
    fig_metrics.update_layout(height=400, margin=dict(l=0, r=0, b=0, t=0), showlegend=False)
    fig_metrics.update_yaxes(title_text="BCE Loss", row=1, col=1)
    fig_metrics.update_yaxes(title_text="||Gradient||", row=2, col=1)
    fig_metrics.update_xaxes(title_text="Epoch", row=2, col=1)
    st.plotly_chart(fig_metrics, width="stretch")

col_a, col_b, col_c = st.columns(3)
col_a.metric("Optimizer", optimizer_choice)
col_b.metric("Final Loss", f"{losses[-1]:.4f}", delta=f"{losses[-1] - losses[0]:.4f}", delta_color="inverse")
col_c.metric("Final Gradient Norm", f"{grads[-1]:.4f}")
