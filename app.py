import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- Configuración del modelo CVAE ---
class CVAE(nn.Module):
    def __init__(self, latent_dim=20, num_classes=10):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(784 + num_classes, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim + num_classes, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x, y):
        x = torch.cat([x, y], dim=1)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        z = torch.cat([z, y], dim=1)
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

# --- Cargar modelo entrenado ---
@st.cache_resource
def load_model():
    model = CVAE()
    model.load_state_dict(torch.load("cvae_mnist.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# --- Interfaz Streamlit ---
st.title("Generador de Dígitos Manuscritos (CVAE)")
st.write("Selecciona un dígito (0–9) para generar 5 imágenes similares.")

digit = st.selectbox("Elige un dígito:", list(range(10)))
generate = st.button("Generar imágenes")

if generate:
    st.subheader(f"Imágenes generadas para el dígito {digit}")

    # Crear etiqueta one-hot
    y = torch.zeros(1, 10)
    y[0, digit] = 1

    cols = st.columns(5)
    for i in range(5):
        z = torch.randn(1, 20)  # espacio latente
        img = model.decode(z, y).detach().numpy().reshape(28, 28)

        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        cols[i].pyplot(fig)
