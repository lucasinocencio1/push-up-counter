# Cross-platform beeps.
# Uses simpleaudio if available; otherwise falls back to '\a' (terminal bell).

import threading

def _beep_sa(freq=880, ms=120):
    try:
        import numpy as np
        import simpleaudio as sa
        sample_rate = 44100
        t = np.linspace(0, ms/1000.0, int(sample_rate*ms/1000.0), False)
        tone = (0.2*np.sin(2*np.pi*freq*t)).astype("float32")
        audio = (tone * 32767).astype("int16")
        play_obj = sa.play_buffer(audio, 1, 2, sample_rate)
        play_obj.wait_done()
    except Exception:
        print("\a", end="")  # silent fallback

def beep(freq=880, ms=120, async_play=True):
    if async_play:
        threading.Thread(target=_beep_sa, args=(freq, ms), daemon=True).start()
    else:
        _beep_sa(freq, ms)
