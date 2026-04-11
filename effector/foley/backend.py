import os

class NullBackend:
    def __init__(self):
        self._log = []
        self._stopped = False
        
    def play(self, filepath, volume, sample_rate):
        self._log.append({"volume": volume, "sample_rate": sample_rate})
        print(f"[Foley] (Muted) Would play: {filepath}")
        
    def play_count(self): return len(self._log)
    def last_play(self): return self._log[-1] if self._log else None
    def stop_all(self): self._stopped = True
    def clear_log(self): self._log.clear()

class PygameBackend:
    def __init__(self):
        import pygame
        # Initialize the mixer for high-quality audio
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.sounds = {}
        
    def play(self, filepath, volume, sample_rate):
        if not filepath or not os.path.exists(filepath):
            print(f"[Foley] Missing audio file: {filepath}")
            return
            
        import pygame
        # Cache the loaded sound so we don't read from the disk every time
        if filepath not in self.sounds:
            try:
                self.sounds[filepath] = pygame.mixer.Sound(filepath)
            except Exception as e:
                print(f"[Foley] Error loading {filepath}: {e}")
                return
                
        sound = self.sounds[filepath]
        sound.set_volume(max(0.0, min(1.0, volume)))
        sound.play()
        
    def stop_all(self):
        import pygame
        pygame.mixer.stop()

def build_backend(options):
    try:
        import pygame
        return PygameBackend()
    except ImportError:
        print("[Foley] Pygame not found. Run 'pip install pygame' to hear audio. Falling back to mute.")
        return NullBackend()