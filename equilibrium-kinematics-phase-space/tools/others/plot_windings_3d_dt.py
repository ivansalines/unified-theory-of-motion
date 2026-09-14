import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ==============================
# PARAMETRI CONFIGURABILI
# ==============================

DATA_FOLDER = "."          # cartella dove stanno i CSV
WINDING_MIN = 1
WINDING_MAX = 20

SORT_MODE = "asc"          # "asc" oppure "desc"
ELEVATION = 20             # inclinazione verticale (gradi)
AZIMUTH = 35               # rotazione orizzontale (gradi)


# ==============================
# CARICAMENTO DATI
# ==============================

all_data = []

for w in range(WINDING_MIN, WINDING_MAX + 1):
    filename = f"summary_winding_{w}_Q1-Q2000.csv"
    filepath = os.path.join(DATA_FOLDER, filename)

    if not os.path.exists(filepath):
        print(f"File non trovato: {filename}")
        continue

    df = pd.read_csv(filepath)

    df["Winding"] = w
    all_data.append(df)

if not all_data:
    raise RuntimeError("Nessun file caricato.")

data = pd.concat(all_data, ignore_index=True)


# ==============================
# ORDINAMENTO
# ==============================

if SORT_MODE == "asc":
    data = data.sort_values(by="Step", ascending=True)
elif SORT_MODE == "desc":
    data = data.sort_values(by="Step", ascending=False)
else:
    raise ValueError("SORT_MODE deve essere 'asc' oppure 'desc'")


# ==============================
# CREAZIONE PLOT 3D
# ==============================

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Separazione per tipo
stiff = data[data["Type"] == "Stiffness"]
conv = data[data["Type"] == "Converged"]

# Scatter 3D
ax.scatter(
    stiff["Winding"],
    stiff["Numberline_Q"],
    stiff["dt"],
    c="black",
    s=5,
    label="Stiffness"
)

ax.scatter(
    conv["Winding"],
    conv["Numberline_Q"],
    conv["dt"],
    c="red",
    s=8,
    label="Converged"
)

# Etichette
ax.set_xlabel("Winding")
ax.set_ylabel("Numberline_Q")
ax.set_zlabel("dt")

# Griglia
ax.grid(True)

# Vista iniziale parametrica
ax.view_init(elev=ELEVATION, azim=AZIMUTH)

plt.legend()
plt.tight_layout()
plt.show()
