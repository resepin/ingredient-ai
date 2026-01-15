# Resepin Ingredient Detection AI Service

A high-performance AI microservice built with **FastAPI** and **YOLOv8** for food ingredient detection. This service is designed to be consumed by a Laravel application via a Dockerized REST API hosted on Azure Web Apps.

## Local Development Setup

To run the application locally for testing before deployment:

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Server**
    ```bash
    uvicorn app.main:app --reload
    ```

3.  **Test the API**
    Open your browser to: `http://127.0.0.1:8000/docs`

---

## Deployment Workflow

This project uses a **Tag-Based Deployment** strategy. Pushing code to the `main` branch will **not** trigger a deployment. The Azure Web App only updates when a new version tag (e.g., v1.0.2) is pushed to GitHub.

### Phase 1: Routine Development (Save Code)
Use this step to save your work to GitHub without affecting the live server.

```powershell
# 1. Stage all changes
git add .

# 2. Commit your changes
git commit -m "Description of your changes"

# 3. Push to GitHub (This will NOT trigger a deploy)
git push origin main
```

### Phase 2: Release to Production (Deploy)
When the code is tested and ready for the live server, use a tag to trigger the GitHub Action.

```powershell
# 1. Create a version tag locally (e.g., v1.0.2)
git tag v1.0.2

# 2. Push the tag to GitHub
# This triggers the "Deploy Release to Azure" workflow
git push origin v1.0.2
```

### Managing Tags
If you make a mistake with a tag, you can delete it:

```powershell
# Delete tag locally
git tag -d v1.0.2

# Delete tag from GitHub
git push --delete origin v1.0.2
```