from __future__ import annotations
from dataclasses import dataclass, field
from statistics import median
from typing import Optional, List

CAUTION_R = 1.10          # R이 이 값 이상이면 최소 CAUTION
WARNING_R = 1.20          # R이 이 값 이상으로 WARNING_HOLD_SEC 이상 유지되면 WARNING
WARNING_HOLD_SEC = 1.5    # WARNING으로 전환되기까지 필요한 연속 유지 시간(초)
RECOVERY_R = 1.10         # WARNING 상태에서 R이 이 값 이하로 내려가야 회복 시작
RECOVERY_HOLD_SEC = 1.0   # SAFE로 복귀(+보상 재생)까지 필요한 연속 유지 시간(초)
MEDIAN_WINDOW = 5         # 순간적인 얼굴 박스 흔들림을 줄이기 위한 최근 R 중앙값 윈도우
CALIBRATION_SEC = 2.0     # 30cm 보정 시 유효 프레임을 모으는 시간(초)
MIN_CALIBRATION_SAMPLES = 10  # 보정 완료에 필요한 최소 유효 샘플 수

SAFE = "SAFE"
CAUTION = "CAUTION"
WARNING = "WARNING"
UNKNOWN = "UNKNOWN"

@dataclass
class DistanceEngine:

    b0: Optional[float] = None              # 보정된 기준 얼굴 면적
    _recent_r: List[float] = field(default_factory=list)
    _warn_timer: float = 0.0                # R>=WARNING_R 연속 유지 시간
    _recover_timer: float = 0.0             # R<=RECOVERY_R 연속 유지 시간(WARNING 중에만 사용)
    persistent_warning: bool = False
    just_rewarded: bool = False             # 이번 프레임에 '까꿍 보상'이 새로 트리거되었는지

    def calibrate(self, areas: List[float]) -> bool:

        if len(areas) < MIN_CALIBRATION_SAMPLES:
            return False
        self.b0 = median(areas)
        self.reset_runtime_state()
        return True

    def reset_runtime_state(self):

        self._recent_r.clear()
        self._warn_timer = 0.0
        self._recover_timer = 0.0
        self.persistent_warning = False
        self.just_rewarded = False

    def update(self, bt: Optional[float], dt: float) -> str:

        self.just_rewarded = False

        if self.b0 is None or bt is None or bt <= 0:

            self._recent_r.clear()
            self._warn_timer = 0.0
            self._recover_timer = 0.0
            return UNKNOWN

        r_raw = (bt / self.b0) ** 0.5
        self._recent_r.append(r_raw)
        if len(self._recent_r) > MEDIAN_WINDOW:
            self._recent_r.pop(0)
        r = median(self._recent_r)

        if self.persistent_warning:

            if r <= RECOVERY_R:
                self._recover_timer += dt
            else:
                self._recover_timer = 0.0

            if self._recover_timer >= RECOVERY_HOLD_SEC:
                self.persistent_warning = False
                self._recover_timer = 0.0
                self._warn_timer = 0.0
                self.just_rewarded = True
                return SAFE
            return WARNING

        if r >= WARNING_R:
            self._warn_timer += dt
            if self._warn_timer >= WARNING_HOLD_SEC:
                self.persistent_warning = True
                self._warn_timer = 0.0
                self._recover_timer = 0.0
                return WARNING
            return CAUTION
        else:
            self._warn_timer = 0.0
            if r >= CAUTION_R:
                return CAUTION
            return SAFE

    @property
    def last_r(self) -> Optional[float]:
        return median(self._recent_r) if self._recent_r else None

def virtual_r_from_distance(distance_cm: float, calib_distance_cm: float = 30.0) -> float:
    return calib_distance_cm / distance_cm

def _simulate():
    engine = DistanceEngine()
    engine.calibrate([100.0] * MIN_CALIBRATION_SAMPLES)  # 30cm 보정 가정 (B0=100)
    print(f"[보정 완료] B0={engine.b0}")

    scenario = (
        [("30cm 유지", 30.0, 2.0)] +
        [("20cm 로 접근", 20.0, 2.0)] +     # R = 30/20 = 1.5 -> WARNING 기대
        [("35cm 로 회복", 35.0, 2.0)]        # R = 30/35 ≈ 0.857 -> SAFE 복귀 + 보상 기대
    )
    t = 0.0
    dt = 0.1
    for label, dist, dur in scenario:
        steps = int(dur / dt)
        for _ in range(steps):
            r = virtual_r_from_distance(dist)
            bt = engine.b0 * (r ** 2)
            state = engine.update(bt, dt)
            t += dt
            if engine.just_rewarded:
                print(f"  t={t:4.1f}s [{label}] R={r:.3f} -> {state}  ★까꿍 보상 재생★")
        print(f"t={t:4.1f}s [{label}] R={virtual_r_from_distance(dist):.3f} -> state={state}")

if __name__ == "__main__":
    _simulate()
