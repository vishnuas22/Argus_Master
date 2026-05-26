"# 19 — Runbook & Operations

> Goal: a single page an operator can open at 2 a.m. when something breaks.
> No prose — every section is a labelled procedure with copy-pasteable
> commands and a clear \"you're done\" condition.
>
> Status: P1 — must exist before First-Finish but can be expanded
> post-launch as new failure modes appear.

---

## 1. First-boot procedure (fresh machine)

Run **once** after a fresh clone:

```bash
# 1. Backend deps
cd /app/backend && pip install -r requirements.txt
# (emergentintegrations must come from the platform wheel; see 01_setup.md §2.2)

# 2. Frontend deps
cd /app/frontend && yarn install

# 3. Mongo running?  (supervisor handles it; sanity-check)
mongosh --eval \"db.adminCommand({ping:1})\"

# 4. ENV present?
test -f /app/backend/.env || cp /app/backend/.env.example /app/backend/.env
test -f /app/frontend/.env || cp /app/frontend/.env.example /app/frontend/.env

# 5. Verify model registry SHAs are reachable
python -m backend.scripts.verify_registry

# 6. Build the reference DB (one-time, ~3 hr at 5000+5000 — see Masterplan §2.1)
python -m backend.scripts.build_reference_db --modalities image --target-per-bucket 5000

# 7. Cold-start calibration (Platt + conformal) — auto-invoked at end of step 6
ls /app/backend/storage/refdb/calibration.json   # must exist
ls /app/backend/storage/refdb/conformal.json     # must exist

# 8. Build the GoldenEval set (~25–45 min)
python -m backend.scripts.eval.download_goldeneval \
    --out /app/backend/storage/eval/goldeneval

# 9. Run a warm-up to preload all detector weights
python -m backend.scripts.warmup

# 10. Smoke test
curl localhost:8001/api/health | jq .   # status: \"ok\", refdb_loaded: true

# 11. Start supervisor (idempotent)
sudo supervisorctl restart backend frontend

# 12. Open http://localhost:3000 — upload a sample
```

**\"You're done\" when:** `/api/health` reports `status=\"ok\"`,
`refdb_loaded=true`, `eval_mini.passed_gate=true`, and a manual upload
returns a verdict within the latency budget.

---

## 2. Warm-up script

```python
# file: /app/backend/scripts/warmup.py
\"\"\"Run a synthetic 64×64 image through the FULL pipeline once.

Forces every detector to load its weights from HF cache → memory.
Without this, the first real user upload blocks for 30–90 s while models
hydrate, which looks like the system is dead.\"\"\"
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from PIL import Image
import io

from backend.services.runner import run_pipeline

log = logging.getLogger(\"warmup\")


async def main() -> None:
    img = Image.new(\"RGB\", (256, 256), (128, 128, 128))
    buf = io.BytesIO(); img.save(buf, format=\"PNG\")
    log.info(\"warmup: starting full-pipeline pass …\")
    result = await run_pipeline(buf.getvalue(), filename=\"warmup.png\")
    log.info(\"warmup: complete  verdict=%s  signals=%d\",
             result[\"verdict\"], len(result[\"signals\"]))


if __name__ == \"__main__\":
    logging.basicConfig(level=logging.INFO, format=\"%(asctime)s %(levelname)s %(name)s %(message)s\")
    asyncio.run(main())
```

**When to run.**
- Once after a fresh deploy.
- Whenever HF cache is wiped (`storage/models/*`).
- On container restart (add to supervisor `program:backend.post_start`).

---

## 3. Reference DB lifecycle

### 3.1 Rebuild cadence

| Reason | When | Risk if not done |
|---|---|---|
| Drift alert sustained > 200 jobs | within 7 days | Accuracy slips on new generators |
| New SOTA generator publicly released | within 30 days | Tier-2 retrieval blind to it |
| Quarterly maintenance | every 90 days | Sources go stale, sample diversity drops |

### 3.2 Rebuild procedure

```bash
# 1. Snapshot existing refDB (safety net)
cp -r /app/backend/storage/refdb /app/backend/storage/refdb.bak.$(date +%Y%m%d)

# 2. Re-run the builder with the same target (additive, not destructive)
python -m backend.scripts.build_reference_db --modalities image --target-per-bucket 5000

# 3. Run cold-start calibration
python -m backend.scripts.run_calibration --modality image --source refdb

# 4. Run GoldenEval (must pass gate before swapping live indexes)
python -m backend.scripts.run_goldeneval \
    --eval-dir /app/backend/storage/eval/goldeneval \
    --report-out /app/backend/storage/eval/reports/refdb_rebuild_$(date +%Y%m%d_%H%M).md

# 5. Restart backend to hot-reload indexes
sudo supervisorctl restart backend

# 6. Confirm
curl localhost:8001/api/refdb/stats | jq .
curl localhost:8001/api/health      | jq '.eval_mini, .drift, .refdb_size'
```

**Rollback:** restore `refdb.bak.<date>` over `refdb/` and restart.

### 3.3 Hard-negative consolidation

The hard-negatives partition grows from user corrections. After ~1000
hard-negatives accumulate (`/api/refdb/stats.image_*_hard > 1000`):

```bash
python -m backend.scripts.consolidate_hard_negatives --modality image
```

Merges hard-neg embeddings into the main FAISS indexes (with re-Platt
+ re-conformal), then empties the hard-neg partition. Idempotent.

---

## 4. API key rotation

### 4.1 Per-provider procedure

| Provider | Rotate via | Time-to-effect |
|---|---|---|
| Gemini (Emergent LLM key) | Emergent dashboard → Universal Key | next call |
| SerpAPI | dashboard.serpapi.com → API key | next call |
| Hive | thehive.ai console → keys → revoke + create | next call |
| SightEngine | dashboard.sightengine.com | next call |
| AI-or-Not | aiornot.com/api/keys | next call |
| HuggingFace | huggingface.co/settings/tokens | next call (re-downloads if cache invalidated) |

### 4.2 Steps (any provider)

```bash
# 1. Generate new key in provider dashboard
# 2. Edit /app/backend/.env — replace OLD with NEW
$EDITOR /app/backend/.env

# 3. Verify no other reference (logs, frontend .env, etc.)
grep -r \"OLD_KEY_PREFIX\" /app/backend /app/frontend  # expect zero hits

# 4. Restart backend (env reloaded only on restart)
sudo supervisorctl restart backend

# 5. Validate
curl localhost:8001/api/health | jq '.gemini_ok, .serpapi_ok'

# 6. Revoke OLD key in provider dashboard
```

**Rule:** never grep keys *into* logs. `_persist` in
`services/usage.py` records provider name and status only — never the key.

---

## 5. Cache management

### 5.1 What is cached, where, and TTL

| Cache | Location | TTL | Safe to clear? |
|---|---|---|---|
| HF model weights | `storage/models/` (`HF_HOME`) | indefinite | Yes — warm-up will re-fetch |
| FAISS indexes | `storage/refdb/*.index` | indefinite | Only with refDB rebuild |
| SerpAPI responses | Mongo `serpapi_cache` | 24 h | Yes — TTL index handles |
| HF Inference API responses | n/a (we don't cache responses) | — | — |
| Frontend asset cache | browser | versioned by build hash | Yes — hard refresh |
| Job artifacts | `storage/jobs/{id}/` | 30 days | Yes — `cleanup_old_jobs.py` |

### 5.2 Clear-all procedure (nuclear)

```bash
# Stops backend
sudo supervisorctl stop backend

# Clear job artifacts older than 30d
python -m backend.scripts.cleanup_old_jobs --days 30

# Clear SerpAPI cache (re-warmable, will cost quota)
mongosh \"$MONGO_URL\" --eval \"db.serpapi_cache.deleteMany({})\"

# DO NOT delete storage/refdb unless rebuilding (see §3.2)
# DO NOT delete storage/models unless rebuilding (warm-up re-downloads)

sudo supervisorctl start backend
python -m backend.scripts.warmup
```

### 5.3 Cleanup script

```python
# file: /app/backend/scripts/cleanup_old_jobs.py
\"\"\"Delete job artifact directories older than N days; preserve Mongo records.\"\"\"
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

ROOT = Path(\"/app/backend/storage/jobs\")


def main(days: int) -> None:
    cutoff = time.time() - days * 86400
    n = 0
    for d in ROOT.glob(\"*\"):
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True); n += 1
    print(f\"cleanup: removed {n} job dirs older than {days}d\")


if __name__ == \"__main__\":
    ap = argparse.ArgumentParser(); ap.add_argument(\"--days\", type=int, default=30)
    main(ap.parse_args().days)
```

---

## 6. Troubleshooting tree

### 6.1 \"Backend won't start\"

```bash
# 1. Check supervisor status
sudo supervisorctl status backend

# 2. Tail logs
tail -200 /var/log/supervisor/backend.err.log

# 3. Common causes:
#    a. Missing env var       → see /app/backend/.env vs .env.example
#    b. Mongo unreachable     → mongosh --eval \"db.adminCommand({ping:1})\"
#    c. Port 8001 in use      → lsof -i :8001
#    d. HF cache corrupt      → rm -rf storage/models && warmup again
#    e. FAISS index corrupt   → restore from storage/refdb.bak.* or rebuild
```

### 6.2 \"First upload hangs 60+ seconds\"

Cause: HF cache cold, models loading on first request.
Fix: run `python -m backend.scripts.warmup` after every container restart.
Long-term: add warm-up to supervisor `post_start` hook.

### 6.3 \"Verdict accuracy dropped suddenly\"

```bash
# 1. Check drift alerts
curl localhost:8001/api/health | jq '.drift.alerts'

# 2. Check eval canary
curl localhost:8001/api/health | jq '.eval_mini'

# 3. Check provider quotas — exhausted provider = missing signal
curl localhost:8001/api/usage | jq '.providers[] | select(.severity != \"ok\")'

# 4. Inspect last 20 INCONCLUSIVE jobs
mongosh \"$MONGO_URL\" --eval '
  db.results.find({verdict:\"INCONCLUSIVE\"}).sort({_id:-1}).limit(20)
    .forEach(r => print(r.signals.map(s => s.name+\":\"+s.p_fake.toFixed(2)).join(\"  \")))
'

# 5. If single signal is consistently extreme → check that detector in
#    Developer Mode for one specific job; tune its weight via 
#    weights_uniform.json or schedule a re-calibration.
```

### 6.4 \"Quota exhausted — what now?\"

| Provider | Behaviour | Action |
|---|---|---|
| Gemini   | VLM signal dropped; narrator → rule-based template | Add balance via Emergent Universal Key dashboard |
| SerpAPI  | Reverse-search signal dropped; banner shown          | Upgrade to paid tier OR wait for month rollover |
| Hive     | Third-party Hive signal dropped                      | Upgrade OR wait |
| SightEngine | Third-party SE signal dropped                     | Upgrade OR wait |
| AI-or-Not  | Third-party AoN signal dropped                     | Upgrade OR wait |

System remains operational on the remaining signals. Expected accuracy
drop with all third-party dropped: ≈ 2–3 % macro AUROC.

### 6.5 \"ECE drift > 0.10 alert\"

```bash
# 1. Confirm with full eval
python -m backend.scripts.run_goldeneval --report-out /tmp/eval_$(date +%H%M).md

# 2. Re-fit Platt + conformal
python -m backend.scripts.run_calibration --modality image --source mixed

# 3. Re-run eval to confirm ECE recovered
python -m backend.scripts.run_goldeneval --report-out /tmp/eval_post_$(date +%H%M).md
diff /tmp/eval_*.md
```

### 6.6 \"OOM on cuda_full (RTX 3050)\"

Cause: LRU eviction failing under concurrency. Mitigation:

```bash
# Cap concurrent jobs to 1 on small VRAM
export MAX_CONCURRENT_JOBS=1
sudo supervisorctl restart backend
```

Add to `.env` permanently if confirmed.

### 6.7 \"FAISS index corrupt / 'Killed' on load\"

Symptom: `MMapError` or process killed at startup.

```bash
# 1. Try restore from snapshot
ls /app/backend/storage/refdb.bak.*

# 2. If no snapshot — rebuild
sudo supervisorctl stop backend
rm /app/backend/storage/refdb/*.index
python -m backend.scripts.build_reference_db --modalities image --target-per-bucket 5000
```

---

## 7. Backup & restore

### 7.1 What to back up

| What | Where | How often | Restore cost |
|---|---|---|---|
| Mongo (jobs, results, labels, usage) | `mongodump` to `storage/backup/` | daily | seconds |
| `storage/refdb/` (indexes, calibration, conformal) | `tar.gz` to `storage/backup/` | weekly OR after every rebuild | minutes |
| `storage/eval/goldeneval/` | one-time; redownloadable | one-time | hours |
| `storage/models/` (HF cache) | n/a; redownloadable | n/a | ~5–10 min |
| `.env` files | secure password manager | on change | manual |

### 7.2 Backup script

```python
# file: /app/backend/scripts/backup.py
\"\"\"Take a consistent backup of Mongo + refDB. Pure stdlib + mongodump.\"\"\"
from __future__ import annotations

import argparse, os, shutil, subprocess, tarfile, time
from pathlib import Path

OUT = Path(\"/app/backend/storage/backup\")
REFDB = Path(\"/app/backend/storage/refdb\")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = time.strftime(\"%Y%m%d_%H%M\")

    # 1. mongodump
    subprocess.check_call([
        \"mongodump\", \"--uri\", os.environ[\"MONGO_URL\"],
        \"-o\", str(OUT / f\"mongo_{ts}\"),
    ])

    # 2. refDB tarball
    with tarfile.open(OUT / f\"refdb_{ts}.tar.gz\", \"w:gz\") as tar:
        tar.add(REFDB, arcname=\"refdb\")

    print(f\"backup complete: {OUT / f'mongo_{ts}'}, {OUT / f'refdb_{ts}.tar.gz'}\")


if __name__ == \"__main__\": main()
```

### 7.3 Restore

```bash
# Mongo
mongorestore --uri \"$MONGO_URL\" --drop /app/backend/storage/backup/mongo_<ts>/

# RefDB
sudo supervisorctl stop backend
rm -rf /app/backend/storage/refdb
tar -xzf /app/backend/storage/backup/refdb_<ts>.tar.gz -C /app/backend/storage/
sudo supervisorctl start backend
```

---

## 8. Deployment notes

### 8.1 Emergent (`cloud_lite`)

- Supervisor manages `backend` (port 8001) + `frontend` (port 3000).
- Hot reload enabled in dev; production mode forces single-worker for now.
- HF cache survives container restart as long as `storage/` volume is
  mounted. Verify with `ls /app/backend/storage/models/` after restart.
- DO NOT pin python version below 3.11.
- Watch for: `OOMKilled` in pod events → upgrade tier OR reduce
  `MAX_CONCURRENT_JOBS`.

### 8.2 Mac (`mac_full`)

- `TORCH_DEVICE=mps` (auto-detected).
- DIRE forced to CPU; benchmarking flag `ENABLE_DIRE_MPS=true` to experiment.
- MongoDB via Homebrew: `brew services start mongodb-community`.
- ffmpeg / libwebp via Homebrew (audio/video phases).

### 8.3 CUDA (`cuda_full`, RTX 3050 4 GB)

- `TORCH_DEVICE=cuda`, fp16 autocast in registry.
- `MAX_CONCURRENT_JOBS=1` mandatory (4 GB VRAM ceiling).
- Sequential model load + `torch.cuda.empty_cache()` between stages.
- Monitor `nvidia-smi -l 1` during first eval — peak should stay < 3 GB.

---

## 9. Disaster recovery

| Scenario | Recovery procedure | RPO | RTO |
|---|---|---|---|
| Mongo data loss | restore from `mongodump` snapshot (§7.3) | ≤ 24 h | ~10 min |
| RefDB corruption | restore `tar.gz` OR rebuild (§3.2) | ≤ 7 days | 10 min (restore) / 3 hr (rebuild) |
| Container vanishes | redeploy from git; warm-up; restore Mongo | n/a | ~30 min |
| All third-party providers offline | system runs on Tier-1 + Tier-2 only; accuracy drop ~3 % | 0 | 0 (graceful) |
| Gemini key revoked | rule-based narrator + no VLM tiebreaker | 0 | 0 (graceful) |

---

## 10. Routine maintenance schedule

| Cadence | Task | Command |
|---|---|---|
| Daily   | Mongo backup | `python -m backend.scripts.backup` (via cron) |
| Daily   | Cleanup old jobs (>30 d) | `python -m backend.scripts.cleanup_old_jobs --days 30` |
| Weekly  | Inspect drift alerts | `curl /api/health | jq '.drift.alerts'` |
| Weekly  | Check provider usage | `curl /api/usage` |
| Monthly | Refresh GoldenEval if manifest_v2 lands | `download_goldeneval.py` |
| Quarterly | Rebuild refDB | §3.2 |
| Quarterly | License audit | `python -m backend.scripts.license_audit` |

---

## 11. Useful one-liners

```bash
# Last 20 jobs with stages timings
mongosh \"$MONGO_URL\" --eval '
  db.results.find().sort({_id:-1}).limit(20).forEach(r =>
    print(r._id, r.verdict, JSON.stringify(r.durations_ms))
  )'

# Total verdict counts this month
mongosh \"$MONGO_URL\" --eval '
  db.results.aggregate([
    {$match:{_id:{$gte:ObjectId.createFromTime(Date.now()/1000 - 30*86400)}}},
    {$group:{_id:\"$verdict\", n:{$sum:1}}}
  ]).forEach(printjson)'

# Hottest signal (highest weight on average)
mongosh \"$MONGO_URL\" --eval '
  db.results.aggregate([
    {$unwind:\"$signals\"},
    {$group:{_id:\"$signals.name\", avg_w:{$avg:\"$signals.weight\"}}},
    {$sort:{avg_w:-1}}, {$limit:10}
  ]).forEach(printjson)'

# Slowest stage (p95)
mongosh \"$MONGO_URL\" --eval '
  db.results.aggregate([
    {$project:{kv:{$objectToArray:\"$durations_ms\"}}},
    {$unwind:\"$kv\"},
    {$group:{_id:\"$kv.k\", p95:{$percentile:{input:\"$kv.v\", p:[0.95], method:\"approximate\"}}}},
    {$sort:{p95:-1}}
  ]).forEach(printjson)'
```

---

## 12. AGENTS.md mapping

| AGENTS.md principle | Where addressed |
|---|---|
| §9 Documentation — runbooks | This file |
| §11 Error handling — graceful degradation | §6.4 quota exhaustion behaviour table |
| §11 Retry mechanisms | Built into providers; covered in `utils/retry.py` |
| §13 Data — backup and recovery | §7 backup/restore + cron schedule |
| §13 Data retention | `cleanup_old_jobs.py` (30-day default) |
| §6 Security — secrets | Key rotation procedure §4 |

---

## 13. Section exit criteria

- [ ] Warm-up script exists, runs in < 90 s on `cloud_lite`
- [ ] `cleanup_old_jobs.py` runs nightly via cron
- [ ] `backup.py` runs nightly via cron
- [ ] Six troubleshooting trees (§6.1–§6.7) are documented
- [ ] Key rotation procedure tested for at least one provider
- [ ] Disaster-recovery table reviewed during PR
"