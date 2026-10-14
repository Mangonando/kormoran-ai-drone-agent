# Hardware Path — DJI Tello (M9)

This document describes how Kormoran is designed to fly a real DJI Tello drone with the same agent code that runs in simulation. M9 is architectured for but not required for the graded build.

## Design Principles

1. **Agent, memory, perception, and UI never import a backend.** Only `cli.py`/config choose `SimDrone | MockDrone | TelloDrone`.
2. **Commands stay relative/body-frame** in metres and degrees. The Tello backend converts to the Tello SDK's centimetre integers.
3. **Pose is an estimate with low confidence.** The Tello has no position sensor — pose is dead-reckoned from commanded moves. Nothing downstream assumes ground truth.
4. **Skills are blocking with timeouts.** Latency and failure are normal results, not exceptions.
5. **Safety limits are strictly tighter on hardware.** Battery floor 30% (vs 15% sim), altitude cap 1.5 m (vs 2.2 m sim), geofence shrunk to room dimensions.

## TelloDrone Implementation (Planned)

```python
class TelloDrone(DroneInterface):
    """djitellopy-based backend."""

    # move_relative → move_forward/back/left/right/up/down (cm, 20-500)
    # rotate → rotate_clockwise/counter_clockwise (degrees)
    # get_frame → get_frame_read().frame (OpenCV BGR → RGB)
    # get_state → battery + dead-reckoned Pose
```

Key adaptations:
- **Unit conversion:** `move(dx_m=0.5)` → `tello.move_forward(50)` (cm)
- **Clamping:** Tello SDK accepts 20–500 cm; smaller moves are clamped up to 20 cm
- **Frame format:** Tello streams H.264 video; `djitellopy` decodes to OpenCV BGR; the backend converts to RGB `Frame`

## Safety Additions for Hardware

| Feature | Implementation |
|---------|---------------|
| Connection heartbeat | Lost Wi-Fi → automatic `land()` |
| Battery floor | 30% (configurable via `configs/tello.yaml`) |
| Altitude cap | 1.5 m hard limit in supervisor |
| Kill switch | Keyboard `Ctrl+C` → `emergency_stop()` → motors off |
| Pre-flight checklist | Documented below; enforced by human operator |
| Propeller guards | Required for all indoor flights |

## Pre-Flight Checklist

- [ ] Propeller guards installed
- [ ] Battery fully charged (> 80%)
- [ ] Wi-Fi connection to Tello verified
- [ ] `kormoran check-tello` passes (battery, frame, state)
- [ ] Flight area clear of people and fragile objects
- [ ] Kill switch operator identified and briefed
- [ ] Geofence bounds set to room dimensions in `configs/tello.yaml`
- [ ] `RECORD_MISSIONS=true` in `.env`

## Perception Adjustments

- **Sim colours are clean; real lighting isn't.** HSV thresholds need per-environment tuning.
- **VLM detector (stretch):** A vision-language model via OpenRouter could replace or augment `ColorDetector` for real-world objects ("blue chair" instead of "blue sphere").

## Sim vs Real Comparison (to be filled after M9)

| Metric | Simulation | Tello (real) |
|--------|-----------|-------------|
| Position accuracy | Ground truth (mm) | Dead-reckoned (±20 cm) |
| Frame rate | 240 Hz (sim) / 30 Hz (camera) | 30 Hz (Wi-Fi stream) |
| Latency per move | ~50 ms | ~500 ms – 2 s |
| Battery drain | Simulated (configurable) | Real (~10 min flight time) |
| Perception quality | Perfect lighting | Variable (shadows, reflections) |
