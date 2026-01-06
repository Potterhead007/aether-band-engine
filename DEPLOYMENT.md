# AETHER Band Engine - Deployment Guide

Production deployment guide for institutional environments.

## Quick Start

### Local Development
```bash
# Install dependencies
pip install -e ".[dev]"

# Run API server
uvicorn src.aether.api.app:app --reload --port 8000

# Run frontend
cd frontend && npm install && npm run dev
```

### Docker
```bash
# Build and run
docker build -t aether-api --target api .
docker run -p 8000:8000 -e AETHER_ENVIRONMENT=production aether-api
```

---

## Production Deployment Options

### 1. Kubernetes (Helm)

```bash
# Add Helm repo and install
helm install aether-engine ./deploy/helm/aether-engine \
  --namespace aether \
  --create-namespace \
  --set image.tag=v1.0.0 \
  --set env.AETHER_ENVIRONMENT=production
```

**Required Secrets:**
```bash
kubectl create secret generic aether-api-secrets \
  --namespace aether \
  --from-literal=ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  --from-literal=OPENAI_API_KEY=$OPENAI_API_KEY
```

### 2. AWS (ECS Fargate)

```bash
cd deploy/aws
terraform init
terraform apply \
  -var="anthropic_api_key=$ANTHROPIC_API_KEY" \
  -var="api_image_tag=v1.0.0" \
  -var="frontend_image_tag=v1.0.0"
```

### 3. GCP (Cloud Run)

```bash
cd deploy/gcp
terraform init
terraform apply \
  -var="project_id=your-project" \
  -var="anthropic_api_key=$ANTHROPIC_API_KEY" \
  -var="api_image_tag=v1.0.0"
```

---

## Environment Variables

### Required
| Variable | Description | Example |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Claude API key | `sk-ant-...` |

### Optional
| Variable | Default | Description |
|----------|---------|-------------|
| `AETHER_ENVIRONMENT` | `development` | `development` or `production` |
| `AETHER_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `AETHER_LOG_FORMAT` | `text` | `text` or `json` |
| `AETHER_API_PORT` | `8000` | API server port |
| `AETHER_API_WORKERS` | `1` | Uvicorn workers |
| `AETHER_LLM_PROVIDER` | `mock` | `mock`, `claude`, `openai` |
| `AETHER_RATE_LIMIT_RPS` | `10` | Requests per second limit |
| `AETHER_AUTH_ENABLED` | `true` | Enable authentication |
| `AETHER_SHUTDOWN_TIMEOUT` | `30` | Graceful shutdown timeout (seconds) |

---

## Health Checks

| Endpoint | Purpose | Expected Response |
|----------|---------|-------------------|
| `GET /health` | Overall health | `200 OK` with component status |
| `GET /ready` | Readiness probe | `200 OK` when ready to serve |
| `GET /live` | Liveness probe | `200 OK` if process is alive |

**Example:**
```bash
curl http://localhost:8000/health
# {"status": "healthy", "checks": {"providers": {"status": "ok"}}}
```

---

## Security Checklist

Before production deployment:

- [ ] Set `AETHER_ENVIRONMENT=production` (hides error details)
- [ ] Configure real LLM API keys (not mock provider)
- [ ] Set up proper CORS origins (not wildcard)
- [ ] Enable authentication (`AETHER_AUTH_ENABLED=true`)
- [ ] Configure TLS/HTTPS termination
- [ ] Set up rate limiting appropriate for your traffic
- [ ] Review and rotate secrets regularly
- [ ] Enable audit logging

---

## Scaling Guidelines

### API Server
- **Minimum replicas:** 3 (for HA)
- **CPU:** 500m request, 2000m limit
- **Memory:** 1Gi request, 4Gi limit
- **HPA:** Scale on CPU (70%) and memory (80%)

### Database Connections
- Use connection pooling
- Max connections per worker: 10
- Total pool size: workers * 10

### Audio Generation
- Audio rendering is CPU-intensive
- Consider dedicated worker pool for `/v1/render` endpoint
- SoundFont files: ~50MB each, cache appropriately

---

## Monitoring

### Metrics Endpoint
```bash
curl http://localhost:8000/metrics
```

### Key Metrics
- `http_requests_total` - Request count by path/method
- `http_request_duration_seconds` - Latency histogram
- `aether_generation_duration_seconds` - Track generation time
- `aether_provider_errors_total` - Provider failure count

### Prometheus ServiceMonitor
```yaml
# Enabled in Helm values
serviceMonitor:
  enabled: true
  interval: 30s
```

---

## Troubleshooting

### API returns 503 Service Unavailable
- Check `/ready` endpoint - providers may not be initialized
- Verify LLM API keys are valid
- Check logs for provider initialization errors

### Slow audio generation
- Verify SoundFont file is available
- Check disk space for temp files
- Consider increasing worker timeout

### Authentication failures
- Verify JWT secret key matches across instances
- Check token expiry settings
- Ensure clock sync between services

### High memory usage
- Audio buffers can be large - set memory limits
- Enable garbage collection logging
- Consider reducing concurrent generations

---

## Support

- **Issues:** https://github.com/aether-band-engine/issues
- **Documentation:** https://docs.aether-engine.com
- **Security:** security@aether-engine.com
