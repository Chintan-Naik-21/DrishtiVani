# Accessible Engineering Notes - Ingestion Service

The **Document Ingestion, Layout Parsing & Database Pipeline** is the foundational stage (Stage 1) of the **Accessible Engineering Notes** system. It takes scanned PDF handouts for **Basic Electrical Circuits**, splits them into pages, detects and classifies layout regions (headings, text, equations, diagrams, tables), parses tabular data into structured row/column grids, runs OCR, and persists traceable metadata in **Supabase** (PostgreSQL + Object Storage). Downstream teammates consume this data for MathML/LaTeX conversion, circuit netlist generation, and teacher review.

---

## Architecture & Tech Stack

- **Framework**: Python 3.11, FastAPI, Uvicorn, Pydantic v2 & `pydantic-settings`
- **Database**: Supabase PostgreSQL managed via **SQLAlchemy 2.0** and **psycopg (v3)**
- **Migrations**: **Alembic** with Row-Level Security (RLS) enabled
- **Object Storage**: Supabase Storage (`supabase-py` client, server-side service-role only) with time-limited signed URLs
- **Page Rasterization**: **PyMuPDF (fitz)** at 150 DPI for optimal storage efficiency
- **Layout & OCR**: **Surya OCR** for layout segmentation, reading order, and text recognition (with optional LayoutLMv3 fallback)
- **Table Parsing**: HuggingFace **Table Transformer (TATR)** (`microsoft/table-transformer-structure-recognition`)
- **Testing**: **pytest** with unit and API integration tests

---

## Supabase Setup Guide

### 1. Create a Supabase Project
1. Log in to [Supabase](https://supabase.com).
2. Click **New Project** and choose your preferred organization and cloud region.
3. Set a strong database password and save it securely.

### 2. Retrieve Supabase API Credentials
1. In your project dashboard, navigate to **Project Settings** (gear icon) -> **API**.
2. Copy the **Project URL**: this is your `SUPABASE_URL`.
3. Scroll to **Project API keys** and locate the **`service_role` (secret)** key.
   - ⚠️ **Important**: Do not use the `anon` key. The ingestion service requires the `service_role` key to create buckets and upload scan assets securely on the backend.
   - Copy this string: this is your `SUPABASE_SERVICE_ROLE_KEY`.

### 3. Retrieve Database Connection Pooler URI
1. Navigate to **Project Settings** -> **Database**.
2. Scroll down to **Connection string** and select the **URI** tab.
3. Switch the mode to **Connection pooler** (Port `6543`, Transaction Mode).
   - *Why connection pooler?* Supabase free-tier and serverless environments recommend the connection pooler to prevent connection exhaustion. The service automatically disables prepared statements (`prepare_threshold=None`) to ensure 100% compatibility with PgBouncer.
4. Replace `[YOUR-PASSWORD]` with your actual database password.
5. Copy this string: this is your `DATABASE_URL`.

### 4. Storage Buckets
You **do not** need to manually create storage buckets in the Supabase UI. On startup, the service automatically creates the three required buckets as **PRIVATE** buckets:
- `handouts`: Stores the original uploaded PDF files.
- `pages`: Stores rendered page raster images (150 DPI PNG/JPEG).
- `crops`: Stores cropped equation and circuit diagram images for downstream services.

> [!NOTE]
> **Free-Tier Project Pausing**: Supabase Free Tier projects automatically pause after a week of inactivity. On startup or when hitting `/health`, the service verifies DB connectivity and will report a clear message directing you to the Supabase dashboard to unpause if the connection fails.

---

## Local Development & Setup

### 1. Clone & Enter Project Directory
```bash
git clone https://github.com/blanijoystan-droid/DrishtiVani.git
cd DrishtiVani/ingestion-service
```

### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```
Open `.env` and fill in your Supabase credentials:
```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJh...your-service-role-key
DATABASE_URL=postgresql://postgres.your-project-id:[YOUR-PASSWORD]@aws-0-us-east-1.pooler.supabase.com:6543/postgres

SUPABASE_BUCKET_HANDOUTS=handouts
SUPABASE_BUCKET_PAGES=pages
SUPABASE_BUCKET_CROPS=crops
RENDER_DPI=150
PAGE_IMAGE_FORMAT=PNG
SIGNED_URL_EXPIRATION_SECONDS=3600
CONFIDENCE_THRESHOLD=0.80
USE_LAYOUTLMV3_FALLBACK=false
SKIP_STARTUP_CHECKS=false
```

### 5. Run Database Migrations
Apply the initial schema, indexes, and Row Level Security (RLS) rules to your Supabase PostgreSQL instance:
```bash
alembic upgrade head
```

To inspect the generated SQL statements without applying them:
```bash
alembic upgrade head --sql
```

### 6. Start the Service
```bash
uvicorn app.main:app --reload --port 8000
```
The interactive API documentation will be available at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Running with Docker (Optional)

You can containerize and run the service via Docker:
```bash
# Build the Docker image
docker build -t accessible-notes-ingestion .

# Run the container with your .env file
docker run --rm -p 8000:8000 --env-file .env accessible-notes-ingestion
```

---

## Testing

Run the automated test suite with pytest:
```bash
pytest -v
```

---

## API Verification & Example Requests

### 1. System Health Check
Verifies both Supabase PostgreSQL connectivity and Supabase Storage reachability:
```bash
curl -X GET http://localhost:8000/health
```
**Healthy Response (HTTP 200)**:
```json
{
  "status": "healthy",
  "database": {
    "status": "connected"
  },
  "storage": {
    "status": "connected",
    "buckets": ["handouts", "pages", "crops"]
  }
}
```

**Paused or Disconnected Project (HTTP 503)**:
```json
{
  "status": "unhealthy",
  "database": {
    "status": "disconnected",
    "error": "Failed to connect to Supabase PostgreSQL database. If using Supabase Free Tier, the project may be paused due to inactivity. Check https://supabase.com/dashboard"
  },
  "storage": {
    "status": "connected"
  }
}
```

### 2. Upload Document (Phase 2)
```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@/path/to/circuits_lecture_notes.pdf"
```

### 3. Check Document Status (Phase 2)
```bash
curl -X GET http://localhost:8000/documents/123e4567-e89b-12d3-a456-426614174000
```

### 4. Fetch Structured Layout JSON (Phase 6)
```bash
curl -X GET http://localhost:8000/documents/123e4567-e89b-12d3-a456-426614174000/layout
```

### 5. Update Block Verification Status (Phase 6)
```bash
curl -X PATCH http://localhost:8000/blocks/234e5678-e89b-12d3-a456-426614174001/verification \
  -H "Content-Type: application/json" \
  -d '{"verification_status": "approved", "verified_by": "prof_sharma"}'
```
