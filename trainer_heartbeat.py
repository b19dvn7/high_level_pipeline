import sys, os, time, json, math, signal
from pathlib import Path

class Heartbeat:
    def __init__(self, outdir: Path, total_steps: int, tag: str="train", refresh=2):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.csv = (self.outdir / "train_log.csv").open("a", buffering=1)
        if self.csv.tell() == 0:
            self.csv.write("ts,step,total,eta_s,secs_per_k,seed,buf,loss_q,loss_pi,loss_bc,td_abs,pi_norm,q1,q2,reward,eps\n")
        self.now = (self.outdir / "now_train.csv")
        self.total = total_steps
        self.t0 = time.time()
        self._last = self.t0
        self.refresh = refresh
        self._last_print = 0
        self.tag = tag
        self._want_stop = False
        signal.signal(signal.SIGINT, self._sigint)

    def _sigint(self, *_):
        self._want_stop = True
        print("\n[heartbeat] SIGINT received — saving state then exiting cleanly...", file=sys.stderr)

    @property
    def want_stop(self): return self._want_stop

    def _eta(self, step):
        elapsed = max(1e-6, time.time() - self.t0)
        rate = step / elapsed
        rem = max(0.0, (self.total - step) / max(1e-6, rate))
        return rem, 1000.0 / max(1e-9, rate)  # secs_per_k

    def log(self, *, step:int, seed:int, buf:int, loss_q:float, loss_pi:float, loss_bc:float,
            td_abs:float, pi_norm:float, q1:float, q2:float, reward:float, eps:float):
        ts = time.time()
        eta_s, spk = self._eta(step)
        self.csv.write(f"{ts:.3f},{step},{self.total},{eta_s:.1f},{spk:.3f},{seed},{buf},{loss_q:.6f},{loss_pi:.6f},{loss_bc:.6f},{td_abs:.6f},{pi_norm:.6f},{q1:.6f},{q2:.6f},{reward:.6f},{eps:.6f}\n")
        # tiny “now” file for other UIs (tail -f)
        self.now.write_text(
            f"step={step}/{self.total}  eta={eta_s/60.0:,.1f}m  secs/1k={spk:.2f}  "
            f"loss_q={loss_q:.4f}  loss_pi={loss_pi:.4f}  bc={loss_bc:.4f}  "
            f"td|={td_abs:.4f}  |pi|={pi_norm:.3f}  q1={q1:.4f} q2={q2:.4f}  "
            f"buf={buf}  seed={seed}\n"
        )
        # periodic single-line console heartbeat (no spam)
        if ts - self._last_print >= self.refresh:
            pct = 100.0 * step / max(1, self.total)
            bar_w, filled = 28, int(28 * pct / 100.0)
            bar = "[" + "#"*filled + "-"*(bar_w-filled) + "]"
            print(
                f"\r{bar} {pct:5.1f}%  step {step}/{self.total}  eta {eta_s/60.0:5.1f}m  "
                f"q1={q1:7.4f} q2={q2:7.4f}  pi={pi_norm:6.3f}  "
                f"Q={loss_q:6.4f} Pi={loss_pi:6.4f} BC={loss_bc:6.4f}  buf={buf:7d}",
                end="",
                file=sys.stdout,
                flush=True
            )
            self._last_print = ts

    def close(self):
        try: self.csv.close()
        except: pass
