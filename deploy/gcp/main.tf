# AETHER Band Engine - GCP Infrastructure
# Terraform configuration for Cloud Run deployment
#
# Usage:
#   terraform init
#   terraform plan -var="anthropic_api_key=$ANTHROPIC_API_KEY" -var="project_id=your-project"
#   terraform apply -var="anthropic_api_key=$ANTHROPIC_API_KEY" -var="project_id=your-project"

terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  backend "gcs" {
    bucket = "aether-terraform-state"
    prefix = "production"
  }
}

# =============================================================================
# Variables
# =============================================================================

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}

variable "anthropic_api_key" {
  description = "Anthropic API key"
  type        = string
  sensitive   = true
}

variable "api_image_tag" {
  description = "API Docker image tag (never use 'latest' in production)"
  type        = string
  default     = "v1.0.0"
}

variable "frontend_image_tag" {
  description = "Frontend Docker image tag (never use 'latest' in production)"
  type        = string
  default     = "v1.0.0"
}

# =============================================================================
# Provider
# =============================================================================

provider "google" {
  project = var.project_id
  region  = var.region
}

# =============================================================================
# Enable APIs
# =============================================================================

resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudbuild.googleapis.com",
  ])

  service            = each.key
  disable_on_destroy = false
}

# =============================================================================
# Artifact Registry
# =============================================================================

resource "google_artifact_registry_repository" "main" {
  location      = var.region
  repository_id = "aether"
  description   = "AETHER container images"
  format        = "DOCKER"

  depends_on = [google_project_service.apis]
}

# =============================================================================
# Secret Manager
# =============================================================================

resource "google_secret_manager_secret" "anthropic_key" {
  secret_id = "anthropic-api-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "anthropic_key" {
  secret      = google_secret_manager_secret.anthropic_key.id
  secret_data = var.anthropic_api_key
}

# =============================================================================
# Service Accounts
# =============================================================================

resource "google_service_account" "api" {
  account_id   = "aether-api"
  display_name = "AETHER API Service Account"
}

resource "google_service_account" "frontend" {
  account_id   = "aether-frontend"
  display_name = "AETHER Frontend Service Account"
}

resource "google_secret_manager_secret_iam_member" "api_secret_access" {
  secret_id = google_secret_manager_secret.anthropic_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.api.email}"
}

# =============================================================================
# Cloud Run - API
# =============================================================================

resource "google_cloud_run_v2_service" "api" {
  name     = "aether-api"
  location = var.region

  template {
    service_account = google_service_account.api.email

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/aether/api:${var.api_image_tag}"

      ports {
        container_port = 8000
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
        cpu_idle = true
      }

      env {
        name  = "AETHER_ENVIRONMENT"
        value = var.environment
      }

      env {
        name  = "AETHER_LOG_FORMAT"
        value = "json"
      }

      env {
        name = "ANTHROPIC_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.anthropic_key.secret_id
            version = "latest"
          }
        }
      }

      startup_probe {
        http_get {
          path = "/health"
          port = 8000
        }
        initial_delay_seconds = 10
        period_seconds        = 3
        failure_threshold     = 10
      }

      liveness_probe {
        http_get {
          path = "/live"
          port = 8000
        }
        period_seconds    = 30
        failure_threshold = 3
      }
    }

    scaling {
      min_instance_count = var.environment == "production" ? 1 : 0
      max_instance_count = 10
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [
    google_project_service.apis,
    google_secret_manager_secret_version.anthropic_key,
  ]
}

# =============================================================================
# Cloud Run - Frontend
# =============================================================================

resource "google_cloud_run_v2_service" "frontend" {
  name     = "aether-frontend"
  location = var.region

  template {
    service_account = google_service_account.frontend.email

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/aether/frontend:${var.frontend_image_tag}"

      ports {
        container_port = 3000
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
        cpu_idle = true
      }

      env {
        name  = "NEXT_PUBLIC_API_URL"
        value = google_cloud_run_v2_service.api.uri
      }
    }

    scaling {
      min_instance_count = var.environment == "production" ? 1 : 0
      max_instance_count = 10
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [
    google_project_service.apis,
    google_cloud_run_v2_service.api,
  ]
}

# =============================================================================
# IAM - Public Access
# =============================================================================

resource "google_cloud_run_v2_service_iam_member" "api_public" {
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_service_iam_member" "frontend_public" {
  location = google_cloud_run_v2_service.frontend.location
  name     = google_cloud_run_v2_service.frontend.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# =============================================================================
# Outputs
# =============================================================================

output "api_url" {
  description = "API Cloud Run URL"
  value       = google_cloud_run_v2_service.api.uri
}

output "frontend_url" {
  description = "Frontend Cloud Run URL"
  value       = google_cloud_run_v2_service.frontend.uri
}

output "artifact_registry" {
  description = "Artifact Registry URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/aether"
}
