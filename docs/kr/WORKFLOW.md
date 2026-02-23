# 개발 워크플로우

이 저장소는 **이중 브랜치 전략**을 사용합니다.

- `private-main`: 연구/운영 전체 코드(전략, 내부 도구, 민감 워크플로우 포함)
- `main`: 공개 배포용 브랜치(핵심 엔진 중심)

## 1. Public 브랜치 제거 대상

`main`에 공개 배포할 때는 DB 구축/동기화 코드를 포함하지 않습니다.

- `lumina_quant/data_sync.py`
- `lumina_quant/data_collector.py`
- `scripts/sync_binance_ohlcv.py`
- `scripts/collect_market_data.py`
- `scripts/collect_universe_1s.py`
- `tests/test_data_sync.py`

정책:

- Public 브랜치는 DB **읽기 전용** 사용만 허용 (기존 SQLite/CSV 소비)
- 거래소 OHLCV 수집/초기 구축 파이프라인은 Private 브랜치에서만 관리
- DB/로그/런타임 산출물은 git에 포함하지 않음

## 2. 자동화 스크립트

### A. Private 저장소 동기화

`private-main`에서 실행:

- Windows: `./sync_private.ps1`
- Mac/Linux: `./sync_private.sh`

### B. Public 저장소 배포

- Windows: `./publish_api.ps1`
- Mac/Linux: `./publish_api.sh`

## 3. 수동 배포 절차

### Private 반영

```bash
git checkout private-main
git merge main
git add .
git commit -m "sync"
git push private private-main:main
```

### Public 반영

```bash
git checkout main
git merge private-main --no-commit --no-ff
git checkout HEAD -- .gitignore
git rm -f lumina_quant/data_sync.py lumina_quant/data_collector.py scripts/sync_binance_ohlcv.py scripts/collect_market_data.py scripts/collect_universe_1s.py tests/test_data_sync.py
git reset
git add .
git commit -m "chore: publish"
git push origin main
```
