# GraphShield - AI-Powered Plagiarism Detection Engine

Enterprise-grade visual and textual plagiarism detection for academic papers and documents.

## Features

- **Visual Analysis**: Perceptual hashing (pHash/dHash) for figure extraction and comparison
- **Text Analysis**: Semantic similarity with sentence transformers + lexical fallback
- **Cross-Document Matching**: Cloud SQL (PostgreSQL) persistence for database-wide comparison
- **Audit Reports**: Executive-grade PDF reports with evidence links
- **Production Ready**: API keys, rate limiting, structured logging, health checks, Docker

## Quick Start

### Local Development (SQLite)

```bash
# Clone and setup
git clone https://github.com/Gurshan-Shergill/graphshield
cd graphshield

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
uvicorn main:app --reload
```

### Production with Cloud SQL

```bash
# Configure
cp .env.example .env
# Edit .env with your Cloud SQL credentials

# Set API key
export API_KEYS="your-secure-random-key"

# Run with Docker
docker-compose up -d
```

## API Usage

### Scan a PDF

```bash
curl -X POST "http://localhost:8000/api/v1/scan-pdf" \
  -H "X-API-Key: your-secure-random-key" \
  -F "file=@paper.pdf"
```

### Download Report

```bash
curl -H "X-API-Key: your-secure-random-key" \
  "http://localhost:8000/api/v1/download-audit-report" \
  --output audit_report.pdf
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_CLOUD_SQL` | Enable Cloud SQL | `false` |
| `CLOUD_SQL_INSTANCE` | Instance connection name | - |
| `DB_USER` | Database user | `postgres` |
| `DB_PASS` | Database password | - |
| `DB_NAME` | Database name | `graphshield` |
| `PRIVATE_IP` | Use private IP | `false` |
| `API_KEYS` | Comma-separated API keys | - |
| `RATE_LIMIT_REQUESTS` | Requests per window | `100` |
| `RATE_LIMIT_WINDOW` | Window in seconds | `60` |
| `LOG_LEVEL` | Log level | `INFO` |
| `LOG_FORMAT` | `json` or `text` | `json` |
| `MAX_FILE_SIZE_MB` | Max upload size | `50` |

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│   Upload    │────▶│  Extraction  │────▶│   Hashing      │
│   (PDF)     │     │  (PyMuPDF)   │     │  (pHash/dHash) │
└─────────────┘     └──────────────┘     └───────┬────────┘
                                                 │
                    ┌──────────────┐     ┌───────▼────────┐
                    │   Database   │◀───▶│  Cross-Ref     │
                    │  (Cloud SQL) │     │  Matching      │
                    └──────────────┘     └───────┬────────┘
                                                 │
                    ┌──────────────┐     ┌───────▼────────┐
                    │   Report     │◀───▶│  Semantic      │
                    │  (ReportLab) │     │  Text Match    │
                    └──────────────┘     └────────────────┘
```

## Database Schema

```sql
CREATE TABLE figure_hashes (
    id SERIAL PRIMARY KEY,
    paper_name TEXT NOT NULL,
    figure_path TEXT NOT NULL,
    phash TEXT NOT NULL,
    dhash TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_figure_hashes_phash ON figure_hashes (phash);
CREATE INDEX idx_figure_hashes_paper ON figure_hashes (paper_name);
```

## Testing

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

## Deployment

### Google Cloud Run

```bash
gcloud run deploy graphshield \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars USE_CLOUD_SQL=true,CLOUD_SQL_INSTANCE=proj:region:inst,DB_PASS=secret
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: graphshield
spec:
  replicas: 3
  selector:
    matchLabels:
      app: graphshield
  template:
    spec:
      containers:
      - name: api
        image: gcr.io/your-project/graphshield:latest
        ports:
        - containerPort: 8000
        envFrom:
        - secretRef:
            name: graphshield-secrets
```

## License

MIT