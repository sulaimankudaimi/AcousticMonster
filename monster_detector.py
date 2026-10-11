"""Sovereign P.12 decision unit -- acoustic blockage detector.

Import-safe: no figures are created, no models trained and no terminal output
happens at import time. Interactive visuals are opt-in via
:class:`MonitorVisuals` and the demo in ``main()``.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_BLOCKAGE_THRESHOLD = 0.7  # mirrors config decision.blockage_threshold


def monster_decision_unit(
    signal, env_data, model, original_packet, threshold=DEFAULT_BLOCKAGE_THRESHOLD
):
    """Analyse a live packet, run inference and update the monitor.

    Args:
        signal (array): Acoustic signal shaped ``(1, 500, 1)``.
        env_data (array): Environmental context shaped ``(1, 3)``.
        model: Compiled Keras model (output: prob, location, confidence).
        original_packet (dict): The full sovereign packet, for telemetry.
        threshold (float): Blockage probability cutoff (default 0.7).

    Returns:
        dict: Decision summary ``{blocked, prob, location, confidence}``.
    """
    try:
        prediction = model.predict([signal, env_data], verbose=0)
        prob_blocked = float(prediction[0][0])
        detected_loc = float(prediction[0][1]) * 500
        confidence = float(prediction[0][2])

        header = original_packet.get("header", {})
        physics = original_packet.get("physics_vector", {})
        financial = original_packet.get(
            "financial_engine", {"current_risk_exposure": 0}
        )

        is_blocked = prob_blocked > threshold
        summary = {
            "blocked": is_blocked,
            "prob_blocked": prob_blocked,
            "location_m": detected_loc,
            "confidence": confidence,
        }

        update_visuals(
            np.asarray(signal).flatten(), prob_blocked, detected_loc, physics
        )
        render_terminal_report(
            header, is_blocked, detected_loc, confidence, financial
        )
        return summary
    except Exception as exc:  # noqa: BLE001 - keep the live stream alive
        logger.error("detector failed to process packet: %s", exc)
        return None


class MonitorVisuals:
    """Lazy, optional interactive monitor (only created when requested)."""

    def __init__(self):
        import matplotlib

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        self.plt = plt
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.canvas.manager.set_window_title(
            "Sovereign P.12 - Acoustic Monster Vision"
        )

    def refresh(self, signal_data, prob, loc, physics):
        import numpy as np

        self.ax1.clear()
        self.ax2.clear()
        self.ax1.plot(signal_data, color="#00ff41", linewidth=1)
        self.ax1.set_title(
            f"Acoustic Pulse Integrity | Pressure: {physics.get('pressure', 0)} Pa",
            color="white",
        )
        self.ax1.set_facecolor("black")
        self.ax1.grid(True, alpha=0.2)

        x_axis = np.linspace(0, 500, 500)
        if prob > 0.7:
            self.ax2.axvline(x=loc, color="red", linestyle="--", label="Potential Blockage")
            self.ax2.fill_between(x_axis, 0, prob, color="red", alpha=0.3)
        else:
            self.ax2.fill_between(x_axis, 0, prob, color="cyan", alpha=0.3)
        self.ax2.set_title(f"Blockage Probability: {prob*100:.1f}%", color="white")
        self.ax2.set_ylim(0, 1)
        self.ax2.set_facecolor("black")
        self.ax2.legend()
        self.plt.tight_layout()
        self.fig.patch.set_facecolor("#121212")
        self.plt.pause(0.01)


_MONITOR = None


def update_visuals(signal_data, prob, loc, physics):
    """Refresh the persistent monitor if it has been initialised, else no-op."""
    if _MONITOR is not None:
        _MONITOR.refresh(signal_data, prob, loc, physics)


def render_terminal_report(header, is_blocked, loc, conf, financial):
    """Log a compact decision report (safety-relevant info at INFO)."""
    if is_blocked:
        logger.info(
            "MONSTER ANALYSIS time=%s | STATUS=BLOCKAGE DETECTED | "
            "location=%.2f m | confidence=%.2f%% | risk=%.2f USD/hr",
            header.get("timestamp", "N/A"),
            loc,
            conf * 100,
            financial.get("current_risk_exposure", 0.0),
        )
    else:
        logger.info(
            "MONSTER ANALYSIS time=%s | STATUS=PIPELINE CLEAR | "
            "confidence=%.2f%%",
            header.get("timestamp", "N/A"),
            conf * 100,
        )


def main():
    """Demo: build the sovereign brain and feed it a synthetic packet."""
    from src.sovereign_brain import build_sovereign_model

    global _MONITOR

    model = build_sovereign_model()
    rng = np.random.RandomState(7)
    _MONITOR = MonitorVisuals()

    signal = rng.normal(0, 0.1, (1, 500, 1)).astype(np.float32)
    env_data = np.array([[45.0, 50.0, 150.0]], dtype=np.float32)

    packet = {
        "header": {"timestamp": "demo"},
        "physics_vector": {"pressure": 50, "velocity": 150, "temperature": 45},
        "financial_engine": {"current_risk_exposure": 12500.0},
    }
    monster_decision_unit(signal, env_data, model, packet)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()