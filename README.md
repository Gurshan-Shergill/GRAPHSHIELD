# GraphShield — AI-Based Image & Graph Plagiarism Detection Platform

**Turnitin for figures.** Upload a research PDF → Get similarity % → Download UGC-compliant report + deplagiarized PDF.

---

## 🎯 What It Does

| Feature | Description |
|---------|-------------|
| **Figure Extraction** | Auto-extracts all figures, graphs, charts from PDFs |
| **Visual Fingerprinting** | pHash/dHash/wHash + CLIP ViT-B/32 embeddings (512-dim) |
| **Similarity Search** | pgvector HNSW + FAISS cache — sub-100ms across millions of figures |
| **Manipulation Detection** | ELA, SIFT copy-move, inversion/flip/rotation reuse, JPEG artifacts |
| **Graph Semantics** | EasyOCR (EN+HI) for axis labels, data values, legends |
| **Scientific Validity** | Domain-aware constraint checking (band gaps, crystallite sizes, XRD peaks, etc.) |
| **Institutional Repository** | Opt-in org repos + Shodhganga (400K+ theses) |
| **UGC-Compliant Reports** | Executive summary, figure-by-figure, side-by-side, methodology |
| **Deplagiarized PDF** | Flagged figures removed/redacted |
| **Async API** | Upload → Poll status → Download results |
| **Multi-tenancy** | Orgs, users, RBAC (org_admin/instructor/researcher/viewer) |
| **Webhooks** | paper.processed, report.ready, high_similarity.alert |

---

## 🏗 Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│   Upload    │────▶│  Extraction  │────▶│  Fingerprint   │
│   (PDF)     │     │  (PyMuPDF)   │     │ (pHash/CLIP)   │
└─────────────┘     └──────────────┘     └───────┬────────┘
                                                 │
                    ┌──────────────┐     ┌───────▼────────┐
                    │   Database   │◀───▶│  Vector Search │
                    │ (PostgreSQL) │     │ (pgvector/FAISS)│
                    └──────────────┘     └───────┬────────┘
                                                 │
                    ┌──────────────┐     ┌───────▼────────┐
                    │   Reports    │◀───▶│  Analysis      │
                    │ (WeasyPrint) │     │ (Forensics/OCR)│
                    └──────────────┘     └────────────────┘
```

**Services (Docker Compose):**
- `postgres` — pgvector/pgvector:pg16 with HNSW indexes
- `redis` — Celery broker + result backend
- `minio` — S3-compatible object storage
- `api` — FastAPI (2 CPU, 4GB)
- `worker-high` — Celery high-priority queue (2 CPU, 4GB)
- `worker-default` — Celery default queue (4 CPU, 8GB)
- `worker-low` — Celery low-priority queue (2 CPU, 2GB)
- `beat` — Celery Beat scheduler

---

## 🚀 Quick Start

### Prerequisites
- Docker 24+ and Docker Compose v2
- 16GB+ RAM recommended (for CLIP + EasyOCR models)

### 1. Clone & Configure
```bash
git clone https://github.com/Gurshan-Shergill/GRAPHSHIELD
cd GRAPHSHIELD

# Configure environment
cp .env.example .env
# Edit .env with your SECRET_KEY and any custom settings
```

### 2. Start Stack
```bash
docker-compose up -d --build
```

### 3. Initialize Database
```bash
docker-compose exec api alembic upgrade head
```

### 4. Verify Health
```bash
curl http://localhost:8000/health
# {"status":"healthy","version":"3.0.0","database":{"total_figures":0,"total_papers":0},"timestamp":"..."}
```

---

## 📡 API Usage

### Register Organization + Admin
```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"org_name": "IIT Delhi", "email": "admin@iitd.ac.in", "password": "secure123"}'
```

### Login → Get JWT
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@iitd.ac.in", "password": "secure123"}'
# Returns: {"access_token": "...", "refresh_token": "..."}
```

### Create API Key (for scripted uploads)
```bash
curl -X POST http://localhost:8000/api/v1/auth/api-keys \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"name": "scan-script", "scopes": ["papers:write", "papers:read", "reports:read"]}'
# Returns: {"id": 1, "name": "scan-script", "key": "gs_abc123...", ...}
```

### Upload PDF for Scanning
```bash
curl -X POST http://localhost:8000/api/v1/papers/upload \
  -H "X-API-Key: gs_abc123..." \
  -F "file=@research_paper.pdf"
# Returns: {"job_id": "uuid", "paper_id": 1, "status": "queued"}
```

### Poll Processing Status
```bash
curl -H "X-API-Key: gs_abc123..." \
  http://localhost:8000/api/v1/papers/1/status
# Returns: {"paper_id": 1, "status": "processing", "progress": 65, "current_step": "vector_search"}
```

### Get Similarity Results
```bash
curl -H "X-API-Key: gs_abc123..." \
  http://localhost:8000/api/v1/papers/1/results
# Returns: {
#   "overall_similarity": 23.5,
#   "risk_level": "MODERATE",
#   "total_figures": 12,
#   "flagged_figures": 3,
#   "figures": [
#     {"index": 3, "page": 5, "similarity": 87.2, "match_type": "EXTERNAL", "source": "Shodhganga: Thesis 4521", "flags": ["ROTATED_REUSE"]}
#   ]
# }
```

### Download UGC Report PDF
```bash
curl -H "X-API-Key: gs_abc123..." \
  http://localhost:8000/api/v1/papers/1/report \
  --output GraphShield_Report.pdf
```

### Download Deplagiarized PDF
```bash
curl -H "X-API-Key: gs_abc123..." \
  http://localhost:8000/api/v1/papers/1/deplagiarized \
  --output paper_deplagiarized.pdf
```

### Search Figures by Image
```bash
curl -X POST http://localhost:8000/api/v1/search/figures \
  -H "X-API-Key: gs_abc123..." \
  -F "file=@figure.png" \
  -F 'scopes=["my_org", "all_opted_in", "shodhganga"]' \
  -F "top_k=10" \
  -F "threshold=0.70"
```

---

## 🔧 Configuration

Key environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | **Required** - JWT signing key | - |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://postgres:postgres@postgres:5432/graphshield` |
| `MINIO_ROOT_USER/PASSWORD` | MinIO credentials | `minioadmin` / `minioadmin` |
| `CLIP_MODEL` | CLIP model variant | `ViT-B-32` |
| `EASYOCR_LANGUAGES` | OCR languages | `en,hi` |
| `MAX_FILE_SIZE_MB` | Max upload size | `100` |
| `SHODHGANGA_SYNC_ENABLED` | Enable thesis sync | `true` |

---

## 🧪 Testing

```bash
# Run backend tests
cd backend
pytest tests/ -v --cov=app

# Lint
ruff check .
mypy app/
```

---

## 📊 Monitoring

- **API Health**: `GET /health`
- **MinIO Console**: `http://localhost:9001` (minioadmin/minioadmin)
- **Flower (Celery)**: Add to compose for task monitoring

---

## 📁 Project Structure

```
GRAPHSHIELD/
├── backend/
│   ├── app/
│   │   ├── api/v1/           # REST endpoints
│   │   ├── core/             # Config, security, DB, Celery, storage
│   │   ├── models/           # SQLAlchemy + pgvector models
│   │   ├── schemas/          # Pydantic request/response
│   │   ├── services/         # Business logic
│   │   │   ├── extraction/   # PDF → figures
│   │   │   ├── fingerprint/  # pHash + CLIP embeddings
│   │   │   ├── vector_search/ # pgvector + FAISS
│   │   │   ├── manipulation/ # ELA, SIFT, flip/rotate
│   │   │   ├── graph_semantics/ # EasyOCR + data extraction
│   │   │   ├── validity/     # Domain constraint checking
│   │   │   ├── ambiguity/    # CLIP domain consistency
│   │   │   ├── repository/   # Shodhganga + org repos
│   │   │   └── reports/      # Jinja2 + WeasyPrint
│   │   ├── workers/          # Celery tasks
│   │   └── db/migrations/    # Alembic migrations
│   ├── tests/
│   ├── Dockerfile
│   └── pyproject.toml
├── docker-compose.yml
├── .github/workflows/ci.yml
├── .env.example
└── README.md
```

---

## 🛡 Security

- JWT access tokens (15 min) + refresh tokens (7 days)
- API keys with scopes and rate limits
- bcrypt password hashing (12 rounds)
- CORS configurable
- Input validation via Pydantic
- SQL injection prevention via SQLAlchemy ORM

---

## 📜 License

MIT License — see [LICENSE](LICENSE)

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Run tests: `pytest tests/ -v`
4. Lint: `ruff check . && mypy app/`
5. Submit PR

---

## 📞 Support

- Issues: [GitHub Issues](https://github.com/Gurshan-Shergill/GRAPHSHIELD/issues)
- Email: gurshan@example.com

---

**Built for Indian academic integrity** — UGC 2018 compliant, Shodhganga integrated, multilingual OCR ready.