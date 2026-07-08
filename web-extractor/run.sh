#!/usr/bin/env bash
set -euo pipefail

# Start the local Firecrawl-compatible web extractor (fastCRW).
# It connects to the existing SearXNG container for /search requests.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Optional overrides via environment:
#   CRW_API_KEY=...           API key for the local service
#   CRW_SEARCH__SEARXNG_URL=...  SearXNG URL inside the Docker network

export CRW_API_KEY="${CRW_API_KEY:-local}"
export CRW_SEARCH__SEARXNG_URL="${CRW_SEARCH__SEARXNG_URL:-http://searxng:8080}"

docker compose up -d

echo "fastCRW is starting on http://localhost:3000"
echo "Test scrape: curl -X POST http://localhost:3000/v1/scrape -H 'Authorization: Bearer ${CRW_API_KEY}' -H 'Content-Type: application/json' -d '{\"url\":\"https://example.com\",\"formats\":[\"markdown\"]}'"
