# Backend

Run the API locally with:

```bash
python3 -m venv ../.venv
source ../.venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Run tests with:

```bash
../.venv/bin/python -m pytest tests -q
```

The root [README.md](/home/robin-hood/sde/README.md) contains the full stack, Docker, auth, and API usage guide.
