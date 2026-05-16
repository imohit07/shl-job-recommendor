# SHL Assessment Recommender

Conversational AI agent for SHL Individual Test Solutions recommendations. Production-ready FastAPI service with enterprise-grade logging, error handling, and security.

**Version:** 2.0.0 (Production Ready)

## Features

✅ **Conversational AI** - Multi-turn dialogue with Groq AI  
✅ **Intelligent Retrieval** - TF-IDF + cosine similarity for catalog search  
✅ **Comprehensive Logging** - Structured logging for monitoring and debugging  
✅ **Production Security** - CORS restrictions, input validation, error handling  
✅ **Health Checks** - Liveness and readiness probes  
✅ **Docker Ready** - Multi-stage Docker build with non-root user  
✅ **Easy Deploy** - Render, Docker, AWS, GCP, Azure support  

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Application                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Server (Production Ready)              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Input Validation & Error Handling                   │   │
│  │  • Message validation (role, content length)         │   │
│  │  • Request rate limiting preparation                 │   │
│  │  • Standardized error responses                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│        ┌──────────────────┼──────────────────┐              │
│        ▼                  ▼                  ▼              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ TF-IDF Catalog│  │ Claude API   │  │ Logging      │     │
│  │  Retrieval   │  │  Integration │  │ & Monitoring │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## API Endpoints

### GET `/health`
Health check with detailed status.

**Response:**
```json
{
  "status": "ok",
  "version": "2.0.0",
  "environment": "production",
  "catalog_items": 45
}
```

### POST `/chat`
Send messages and receive recommendations.

**Request:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "I am hiring a Java developer"
    }
  ]
}
```

**Response:**
```json
{
  "reply": "To recommend the right assessment...",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://...",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

## Quick Start

### Prerequisites
- Python 3.12+
- Groq API key from [console.Groq.com](https://console.groq.com).
### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variable:**
   ```bash
   export Groq_API_KEY=sk-ant-xxxxxxxxxxxxx
   ```

3. **Start server:**
   ```bash
   python main.py
   ```

4. **Test:**
   ```bash
   # Health check
   curl http://localhost:8000/health
   
   # Chat example
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "I am hiring a Python developer"}]}'
   ```

### Docker

1. **Build image:**
   ```bash
   docker build -t shl-recommender:latest .
   ```

2. **Run container:**
   ```bash
   docker run -p 8000:8000 \
     -e Groq_API_KEY="sk-ant-..." \
     -e ENV="production" \
     shl-recommender:latest
   ```

3. **Or use Docker Compose:**
   ```bash
   Groq_API_KEY=sk-ant-... docker-compose up
   ```

## Production Deployment

⚠️ **Important:** See [PRODUCTION.md](PRODUCTION.md) for comprehensive deployment guidelines.

### Quick Deploy to Render

1. Push to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set environment variable: `Groq_API_KEY=sk-ant-...`
5. Set `ENV=production`
6. Deploy

### Configuration

Configure via environment variables (see [.env.example](.env.example)):

| Variable | Default | Notes |
|----------|---------|-------|
| `Groq_API_KEY` | - | **Required** API key |
| `ENV` | development | Set to `production` for production |
| `PORT` | 8000 | Server port |
| `HOST` | 0.0.0.0 | Bind address |
| `CORS_ORIGINS` | localhost:3000,... | Comma-separated allowed origins |
| `LOG_LEVEL` | INFO | DEBUG, INFO, WARNING, ERROR, CRITICAL |

## Testing

### Manual Testing
```bash
./test-production.sh http://localhost:8000
```

### Load Testing
```bash
# Apache Bench: 1000 requests, 10 concurrent
ab -n 1000 -c 10 http://localhost:8000/health
```

## Monitoring

### Logs
Logs are output to stdout in structured format:
```
2025-05-16 10:30:00 - main - INFO - Loaded catalog with 45 items
2025-05-16 10:30:05 - main - INFO - Chat response: 3 recommendations, eoc=false
```

### Metrics to Monitor
- **Health Check**: Endpoint availability
- **Chat Latency**: Response time percentiles (p50, p95, p99)
- **Error Rate**: 4xx and 5xx response rates
- **Catalog Size**: Number of available assessments
- **API Quota**: Groq API usage

### Example Monitoring Setup
```bash
# CloudWatch (AWS)
docker run --log-driver awslogs \
  --log-opt awslogs-group=/ecs/shl-recommender \
  shl-recommender:latest
```

## Security

✅ **CORS Restriction** - Only allowed origins  
✅ **Input Validation** - Size limits, type checking  
✅ **Non-root Docker** - Runs as unprivileged user  
✅ **Error Sanitization** - No sensitive data in responses  
✅ **API Key Security** - Use environment variables, not .env files in production  

## Troubleshooting

### Service fails to start
```bash
# Check API key is set
echo $Groq_API_KEY

# Check logs
docker logs <container-id>
```

### CORS errors in frontend
Update `CORS_ORIGINS` environment variable:
```bash
CORS_ORIGINS="https://frontend.com,https://api.com" python main.py
```

### High latency
1. Check [status.Groq.com]
2. Add caching layer (Redis)
3. Use load balancer with multiple instances

## Dependencies

- `fastapi>=0.115.0` - Web framework
- `uvicorn>=0.31.0` - ASGI server
- `Groq>=0.39.0` - Groq API client
- `scikit-learn>=1.6.1` - ML for catalog retrieval
- `numpy>=1.26.4` - Numerical computing
- `pydantic>=2.9.2` - Data validation

See [requirements.txt](requirements.txt) for pinned versions.

## Contributing

1. Create feature branch
2. Test locally: `python main.py`
3. Run test suite: `./test-production.sh`
4. Commit and push
5. Create pull request

## License

Proprietary - SHL Group

## Support

- **Documentation**: [PRODUCTION.md](PRODUCTION.md)
- **Issues**: GitHub Issues
- **API Docs (interactive)**: Visit `/docs` when server is running

---

**Version:** 2.0.0  
**Last Updated:** May 16, 2026  
**Status:** Production Ready ✅
