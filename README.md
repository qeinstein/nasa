# 🌌 Exoplanet Dashboard 🚀

The **Exoplanet Dashboard** is a web application designed to explore, predict, and visualize exoplanet data from NASA’s KOI and TESS catalogs. It leverages a **hybrid ensemble model** combining classical machine learning and quantum-inspired models, delivering high-quality predictions, interpretability, and interactive visualizations.

---

## 📌 Features

- **Hybrid Ensemble Predictions:** Combines Random Forest (classical) and Quantum Hybrid (QML) models for robust exoplanet classification.
- **Interactive Visualizations:** Histograms, pairplots, t-SNE, UMAP embeddings, and feature importance plots.
- **Explainability:** SHAP for classical models, permutation importance for QML models, and combined importance plots.
- **Base64 Image Rendering:** All plots are sent to the frontend as Base64 images for seamless rendering.
- **File Upload Support:** Users can upload CSV files to get instant predictions and visualizations.
- **Responsive Dashboard:** Built with React, TailwindCSS, and Framer Motion for smooth animations and responsiveness.

---

## 🛠 Tech Stack

| Layer       | Technology / Libraries |
|------------|-----------------------|
| Frontend   | React, TailwindCSS, Framer Motion, Recharts, Axios, Lucide React |
| Backend    | Python, Flask/FastAPI, PyTorch, PennyLane, scikit-learn, SHAP, UMAP, TSNE, Joblib |
| Data       | NASA KOI & TESS datasets |
| ML Models  | Random Forest Classifier (Classical), Quantum Hybrid Model (QML), Weighted Ensemble |

---

## 🔧 Setup & Installation

### Frontend

```bash
# Clone the repository
git clone <your-repo-url>
cd exoplanet-dashboard

# Install dependencies
npm install

# Start development server
npm run dev