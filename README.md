# Resepin Ingredient Detection AI Service

A high-performance AI microservice built with **FastAPI** and **YOLOv8** for food ingredient detection. This service is designed to be consumed by a Laravel application via a Dockerized REST API.

---

## The Manual Update & Patch Workflow


Step A: Build the Image

```shell
docker build -t ingredient-ai .
```

Step B: Tag for Azure (X: major patch, Y: minor patch)

```shell
docker tag ingredient-ai resepin.azurecr.io/ingredient-ai:vX.Y
```

Step C: Push to Registry

```shell
az acr login --name resepin
docker push resepin.azurecr.io/ingredient-ai:vX.Y
```

Step D: Switch the Tag in Azure

```shell
1. Go to the Azure Portal.
2. Open your App Service (e.g., resepin-api).
3. Go to Deployment Center > Settings.
4. Change the Tag value to the one you just pushed (e.g., v1.1).
5. Click Save.
```