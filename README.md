# Resepin — Ingredient Detection AI Service

A **FastAPI + YOLOv8** microservice that detects food ingredients from images.  
This is the AI inference back-end for the **Resepin** application; the front-end (Laravel) sends an image to the `/predict` endpoint and receives a list of detected ingredient names.

> **Live deployment:** This service is currently running on **Azure App Service** (Web App for Containers) and is consumed by the Laravel front-end at `resepin.azurewebsites.net`.

---

## Table of Contents

1. [Tech Stack](#tech-stack)  
2. [Project Structure](#project-structure)  
3. [Local Setup (Run Without Docker)](#local-setup-run-without-docker)  
4. [Local Setup (Run With Docker)](#local-setup-run-with-docker)  
5. [API Usage](#api-usage)  
6. [Deploying to Azure](#deploying-to-azure)  

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | FastAPI |
| AI Model | YOLOv8 (Ultralytics) — exported to ONNX for faster CPU inference |
| Runtime | Gunicorn + Uvicorn workers |
| Container | Docker |
| CI/CD | GitHub Actions (tag-based deployment) |
| Cloud | Azure App Service (Web App for Containers) + Azure Container Registry |
| Monitoring | Azure Application Insights (OpenTelemetry) |

---

## Project Structure

```
ingredients-api/
├── app/
│   ├── main.py          # FastAPI app, CORS, health-check, model warmup
│   ├── routes.py         # POST /predict endpoint
│   ├── schemas.py        # Pydantic response model
│   └── services.py       # YOLO inference logic & metrics
├── models/
│   ├── best.pt           # YOLOv8 trained weights (PyTorch)
│   └── last.pt           # Last checkpoint
├── .github/
│   └── workflows/
│       └── deploy.yml    # GitHub Actions CI/CD pipeline
├── Dockerfile            # Multi-stage: installs deps → exports ONNX → runs Gunicorn
├── requirements.txt
└── README.md
```

---

## Local Setup (Run Without Docker)

### Prerequisites

- **Python 3.10+**
- **pip** (comes with Python)
- **Git**

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/ingredients-api.git
cd ingredients-api

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the development server
uvicorn app.main:app --reload
```

The server will start at **http://127.0.0.1:8000**.  
Open **http://127.0.0.1:8000/docs** in your browser to access the interactive Swagger UI.

> **Note:** When running locally without Docker, the model is loaded from `models/best.pt` (PyTorch format) by default. The Docker build automatically converts it to ONNX for better performance.

---

## Local Setup (Run With Docker)

Running with Docker mirrors the exact production environment.

### Prerequisites

- **Docker Desktop** — [Download here](https://www.docker.com/products/docker-desktop/)

### Steps

```bash
# 1. Clone the repository (skip if already done)
git clone https://github.com/<your-username>/ingredients-api.git
cd ingredients-api

# 2. Build the Docker image
#    This installs dependencies, exports the model to ONNX, and configures Gunicorn.
docker build -t ingredients-api .

# 3. Run the container
docker run -p 8000:80 ingredients-api
```

The API is now accessible at **http://localhost:8000**.  
Open **http://localhost:8000/docs** for the Swagger UI.

### What the Dockerfile does (behind the scenes)

1. Installs system libraries needed by OpenCV (`libgl1`, `libglib2.0-0`).  
2. Installs CPU-only PyTorch (keeps the image small — no GPU drivers needed).  
3. Installs all Python dependencies from `requirements.txt`.  
4. Copies the app code and model weights into the container.  
5. **Exports the YOLOv8 `.pt` model to `.onnx`** format for ~30–50% faster CPU inference.  
6. Starts a **Gunicorn** server with 2 Uvicorn workers on port 80.

---

## API Usage

### Health Check

```
GET /
```

Response:
```json
{ "status": "Healthy", "version": "1.3.3" }
```

### Predict Ingredients

```
POST /predict
Content-Type: multipart/form-data
```

Upload an image file (field name: `file`).

**Example with curl:**

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@your_food_image.jpg"
```

**Response:**

```json
{
  "ingredients": ["tomato", "onion", "garlic", "chili"]
}
```

---

## Deploying to Azure

This section explains how the project is deployed to **Azure App Service** using **Docker** and **GitHub Actions**.

### Architecture Overview

```
GitHub (push tag) → GitHub Actions → Build Docker Image
    → Push to Azure Container Registry (ACR)
    → Deploy to Azure App Service (Web App for Containers)
```

### Azure Resources Needed

| Resource | Purpose |
|----------|---------|
| **Azure Container Registry (ACR)** | Stores the Docker images |
| **Azure App Service (Linux, B3 SKU)** | Runs the container |
| **Application Insights** *(optional)* | Monitoring & telemetry |

### Step-by-Step: Set Up Azure Resources

#### 1. Create an Azure Container Registry

```bash
# Login to Azure CLI
az login

# Create a resource group (if you don't have one)
az group create --name <your-resource-group> --location southeastasia

# Create the container registry
az acr create --resource-group <your-resource-group> \
  --name <yourRegistryName> --sku Basic --admin-enabled true
```

#### 2. Create an Azure App Service (Web App for Containers)

```bash
# Create an App Service Plan (B3 recommended for this workload)
az appservice plan create --name <your-plan-name> \
  --resource-group <your-resource-group> --sku B3 --is-linux

# Create the Web App pointing to your ACR image
az webapp create --resource-group <your-resource-group> \
  --plan <your-plan-name> --name <your-app-name> \
  --deployment-container-image-name <yourRegistryName>.azurecr.io/ingredient-ai:latest
```

#### 3. Configure GitHub Secrets

In your GitHub repo, go to **Settings → Secrets and variables → Actions** and add:

| Secret | Value |
|--------|-------|
| `ACR_LOGIN_SERVER` | `<yourRegistryName>.azurecr.io` |
| `ACR_USERNAME` | ACR admin username (find in Azure Portal → ACR → Access keys) |
| `ACR_PASSWORD` | ACR admin password |
| `AZURE_WEBAPP_PUBLISH_PROFILE` | Download from Azure Portal → App Service → Get publish profile |

### Deployment Workflow (CI/CD)

This project uses a **tag-based deployment** strategy.  
Pushing to `main` does **not** trigger a deploy — only pushing a version tag does.

#### Push Code (No Deploy)

```bash
git add .
git commit -m "your changes"
git push origin main
```

#### Deploy a New Release

```bash
# Create a version tag
git tag v1.0.2

# Push the tag — this triggers the GitHub Actions pipeline
git push origin v1.0.2
```

The GitHub Actions workflow (`.github/workflows/deploy.yml`) will:

1. Build the Docker image.  
2. Push it to Azure Container Registry with the tag (e.g., `ingredient-ai:v1.0.2`).  
3. Deploy the new image to Azure App Service.

#### Delete a Tag (If Needed)

```bash
git tag -d v1.0.2                     # delete locally
git push --delete origin v1.0.2       # delete from GitHub
```