# 설계 노트: GPU Auto/Fallback + 결정론적 Polars 파이프라인

## 목표

기본 실행 방향은 GPU로 두되, GPU가 없는 환경에서도 동작이 예측 가능하고 재현 가능하도록 유지합니다.

## Compute 해석 계약

`compute_engine.resolve_compute_engine(mode, device, verbose)`는
명시적 인자 또는 환경 변수(`LQ_GPU_MODE`, `LQ_GPU_DEVICE`, `LQ_GPU_VRAM_GB`, `LQ_GPU_VERBOSE`)를
사용해 실행 엔진을 결정합니다.

해석 정책:

1. `cpu`
   - GPU probe를 수행하지 않습니다.
   - 항상 CPU 엔진을 사용합니다.
2. `gpu` / `forced-gpu`
   - GPU smoke probe를 수행합니다.
   - probe가 실패하면 `GPUNotAvailableError`를 발생시킵니다.
3. `auto`
   - 동일한 probe를 수행합니다.
   - 성공하면 GPU, 실패하면 이유를 남기고 CPU로 fallback합니다.

현재 프로젝트 기본값은 **GPU-first**입니다:

- runtime config 기본값은 `execution.gpu_mode=gpu`
- `select_engine()`는 명시적 모드/환경 변수가 없으면 `LQ_GPU_MODE=gpu`를 기본으로 사용합니다
- 일반 non-GPU CI lane은 명시적으로 `LQ_GPU_MODE=cpu`를 override합니다

## 안전 불변식

- **forced 모드에서는 조용한 downgrade가 없어야 합니다**
- **auto 모드는 항상 유효한 엔진을 반환해야 합니다**
- **verbose 모드에서는 요청 모드, 최종 해석 모드, fallback 이유가 보여야 합니다**

## 결정론적 데이터 처리 규칙

CPU/GPU parity를 안정적으로 유지하기 위해:

- 리샘플링은 명시적 bucket 연산(`timestamp_ms // bucket_ms`)을 사용합니다
- `first/last` 집계 전에 timestamp 정렬을 유지합니다
- 핵심 집계 경로에서 `group_by_dynamic` 및 Python UDF를 피합니다
- 출력은 결정론적 제약 조건을 키로 하여 idempotent하게 씁니다

## 상태 idempotency 결합

PostgreSQL 상태 저장은 `run_id`, `dedupe_key`, `fingerprint` 같은 결정론적 conflict key와
`ON CONFLICT DO UPDATE`를 사용하여 replay/retry 중 의미상 중복 row 생성을 방지합니다.

## 운영 가이드

- GPU 우선 로컬 실행에서는 `LQ_GPU_MODE=gpu`를 기본으로 사용합니다
- 공유 장비/혼합 환경에서 opportunistic fallback이 필요하면 `LQ_GPU_MODE=auto`를 사용합니다
- GPU 가용성이 보장되는 환경에서만 `LQ_GPU_MODE=forced-gpu`를 사용합니다
- GPU 활성화 전 extras 설치:
  - `uv sync --extra gpu` (`cudf-polars-cu12`, `nvidia-nvjitlink-cu12` 포함)
- 파이프라인/집계 변경 뒤에는 determinism 테스트를 다시 수행합니다

## CI 설계

CI는 두 단계 GPU 검증 구조를 사용합니다.

### 1. GPU contract job (항상 실행)

표준 `ubuntu-latest` 러너에서 항상 실행합니다.

- GPU extras 설치
- `tests/test_compute_engine.py` 실행
- `tests/test_verify_polars_gpu_runtime_script.py` 실행
- `scripts/ci/verify_polars_gpu_runtime.py`를 skip-safe 모드로 실행
- 실제 GPU 하드웨어가 없어도 코드/의존성/skip semantics가 깨지지 않는지 검증

### 2. Strict GPU runtime smoke (GPU 러너가 있을 때만 실행)

GPU 러너가 설정된 경우에만 실행합니다.

- 저장소 변수 `LQ_GPU_CI_RUNS_ON_JSON`으로 활성화
- `runs-on`에 들어갈 JSON 문자열 또는 JSON label 배열을 기대
- self-hosted 예시:
  - `["self-hosted", "linux", "x64", "gpu"]`
- 선택적 강제 가드:
  - `LQ_GPU_CI_REQUIRED=true`
- `scripts/ci/verify_polars_gpu_runtime.py --require-gpu --mode forced-gpu` 실행
- 엄격한 `polars.GPUEngine` 경로를 사용하며 실제 GPU 실행에 실패하면 job도 실패

이 구조 덕분에 기본 CI는 일반 hosted runner에서도 계속 통과할 수 있고,
전용 NVIDIA 러너에서는 실제 Polars GPU 런타임 검증까지 수행할 수 있습니다.
