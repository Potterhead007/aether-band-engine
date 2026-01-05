# AETHER Deployment Guide

This directory contains deployment configurations for multiple cloud platforms and orchestration systems.

## Quick Start

### Local Development (Docker Compose)

```bash
# Start full stack with mock LLM
docker-compose up -d

# Start with real LLM provider
ANTHROPIC_API_KEY=sk-ant-xxx LLM_PROVIDER=claude docker-compose up -d

# View logs
docker-compose logs -f aether-api

# Access services
# - API: http://localhost:8000
# - Frontend: http://localhost:3001
# - Grafana: http://localhost:3000
# - Prometheus: http://localhost:9090
# - Jaeger: http://localhost:16686
```

## Cloud Deployments

### AWS (ECS Fargate)

```bash
cd deploy/aws

# Initialize Terraform
terraform init

# Plan deployment
terraform plan \
  -var="anthropic_api_key=$ANTHROPIC_API_KEY" \
  -out=tfplan

# Apply
terraform apply tfplan

# Get outputs
terraform output
```

**Architecture:**
- ECS Fargate (serverless containers)
- Application Load Balancer
- ECR for container images
- Secrets Manager for API keys
- CloudWatch for logging
- VPC with public/private subnets

### GCP (Cloud Run)

```bash
cd deploy/gcp

# Initialize Terraform
terraform init

# Plan deployment
terraform plan \
  -var="project_id=your-gcp-project" \
  -var="anthropic_api_key=$ANTHROPIC_API_KEY" \
  -out=tfplan

# Apply
terraform apply tfplan

# Get outputs
terraform output
```

**Architecture:**
- Cloud Run (serverless containers)
- Artifact Registry for images
- Secret Manager for API keys
- Automatic HTTPS with managed certificates

### Kubernetes

```bash
cd deploy/k8s

# Create namespace and secrets
kubectl apply -f namespace.yaml
kubectl create secret generic aether-secrets \
  --from-literal=anthropic-api-key=$ANTHROPIC_API_KEY \
  -n aether

# Deploy application
kubectl apply -f api-deployment.yaml
kubectl apply -f frontend-deployment.yaml
kubectl apply -f ingress.yaml

# Check status
kubectl get pods -n aether
kubectl get hpa -n aether
```

**Features:**
- HorizontalPodAutoscaler for auto-scaling
- Pod anti-affinity for high availability
- Liveness/readiness/startup probes
- Resource limits and requests
- Service accounts with minimal permissions

## Building Images

### API

```bash
# Build
docker build -t aether-api --target api .

# Tag and push to ECR (AWS)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_URL
docker tag aether-api:latest $ECR_URL/aether-api:latest
docker push $ECR_URL/aether-api:latest

# Tag and push to Artifact Registry (GCP)
gcloud auth configure-docker us-central1-docker.pkg.dev
docker tag aether-api:latest us-central1-docker.pkg.dev/$PROJECT_ID/aether/api:latest
docker push us-central1-docker.pkg.dev/$PROJECT_ID/aether/api:latest
```

### Frontend

```bash
cd frontend

# Build
docker build -t aether-frontend .

# Push to registry (same process as API)
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | For Claude | Anthropic API key |
| `OPENAI_API_KEY` | For GPT | OpenAI API key |
| `LLM_PROVIDER` | No | `claude`, `openai`, or `mock` (default: mock) |
| `AETHER_ENVIRONMENT` | No | `development` or `production` |
| `AETHER_LOG_FORMAT` | No | `text` or `json` |
| `REDIS_URL` | No | Redis connection string |

## Monitoring

All deployments include:
- **Health endpoints:** `/health`, `/ready`, `/live`
- **Metrics:** `/metrics` (Prometheus format)
- **Logging:** Structured JSON logs in production

For Kubernetes, ensure Prometheus is configured to scrape the annotations:
```yaml
prometheus.io/scrape: "true"
prometheus.io/port: "8000"
prometheus.io/path: "/metrics"
```

## Security Considerations

1. **API Keys:** Always use secret management (AWS Secrets Manager, GCP Secret Manager, K8s Secrets)
2. **Network:** API runs in private subnets with ALB/Ingress for public access
3. **Container:** Non-root user, minimal base image
4. **TLS:** Configure HTTPS at the load balancer level
