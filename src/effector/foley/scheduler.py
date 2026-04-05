import threading
import time

class ScheduleHandle:
    def __init__(self, event, cancel_cb):
        self.event = event
        self._cancel_cb = cancel_cb
        self.cancelled = False
    def cancel(self):
        self.cancelled = True
        self._cancel_cb(self)

class FoleyScheduler:
    def __init__(self, emitter, config):
        self.emitter = emitter
        self.max_pending = config.get("max_pending", 10)
        self.pending = []
        self.lock = threading.Lock()
        self._stats = {"total_scheduled": 0, "total_fired": 0, "total_cancelled": 0}

    def schedule(self, delay_ms, event):
        with self.lock:
            if len(self.pending) >= self.max_pending: return None
            handle = ScheduleHandle(event, self._remove_handle)
            self.pending.append(handle)
            self._stats["total_scheduled"] += 1

        def _fire():
            time.sleep(delay_ms / 1000.0)
            if not handle.cancelled:
                try: self.emitter(event)
                except Exception: pass
                with self.lock:
                    if handle in self.pending:
                        self.pending.remove(handle)
                        self._stats["total_fired"] += 1

        threading.Thread(target=_fire, daemon=True).start()
        return handle

    def schedule_sequence(self, steps):
        return [self.schedule(d, e) for d, e in steps]

    def _remove_handle(self, handle):
        with self.lock:
            if handle in self.pending:
                self.pending.remove(handle)
                self._stats["total_cancelled"] += 1

    def cancel_all(self):
        with self.lock:
            count = len(self.pending)
            for h in list(self.pending): h.cancel()
            return count

    def pending_count(self):
        with self.lock: return len(self.pending)

    def stats(self):
        with self.lock: return dict(self._stats)
