import pygame as pyg 
import cv2
import sys
import random
from ModulPose import ClientPose
import os

# Konfigurasi TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

GROUND_Y = 650
GRAVITY = 1500

def load_animation_frames(path):
    if not os.path.exists(path):
        print(f"Warning: Path {path} not found.")
        return []
    files = sorted([f for f in os.listdir(path) if f.endswith(".png")], 
                   key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    frames = []
    for f in files:
        img = pyg.image.load(os.path.join(path, f)).convert_alpha()
        frames.append(img)
    return frames

class Animation:
    def __init__(self, frames, frame_rate=0.1, loop=True):
        self.frames = frames
        self.frame_rate = frame_rate
        self.loop = loop
        self.current_frame = 0
        self.timer = 0
        self.finished = False

    def update(self, dt):
        if not self.frames: return
        if self.finished and not self.loop:
            return
        self.timer += dt
        if self.timer >= self.frame_rate:
            self.timer = 0
            self.current_frame += 1
            if self.current_frame >= len(self.frames):
                if self.loop:
                    self.current_frame = 0
                else:
                    self.current_frame = len(self.frames) - 1
                    self.finished = True

    def get_frame(self):
        if not self.frames: return None
        return self.frames[self.current_frame]

    def reset(self):
        self.current_frame = 0
        self.timer = 0
        self.finished = False

# ==========================================
# 1. CLASS PARENTS: SPRITE, LAYAR, JURUS
# ==========================================

class Sprite:
    """Class Parent untuk semua entitas yang memiliki HP dan Speed"""
    def __init__(self, x, y, hp, speed):
        self.pos_x = float(x)
        self.pos_y = float(y)
        self.hp = hp
        self.speed = speed
        self.rect = pyg.Rect(int(x), int(y), 40, 40) # Default rect

    def update(self, dt):
        pass

    def draw(self, screen):
        pass

class Layar:
    """Class Parent untuk manajemen tampilan layar"""
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def draw_text(self, screen, text, font, color, x, y, center=True):
        img = font.render(text, True, color)
        rect = img.get_rect()
        if center: rect.center = (x, y)
        else: rect.topleft = (x, y)
        screen.blit(img, rect)

class Jurus:
    """Class Parent untuk mekanisme serangan / skill"""
    def __init__(self, speed, damage):
        self.speed = speed
        self.damage = damage

# ==========================================
# 2. INHERITANCE: MAIN CHARACTER & BOSS
# ==========================================

class Peluru(Jurus):
    """Inheritance dari Jurus (Mekanisme Peluru)"""
    def __init__(self, x, y, direction, speed=600.0, damage=3):
        super().__init__(speed, damage)
        self.pos_x = float(x)
        self.pos_y = float(y)
        self.direction = direction
        self.rect = pyg.Rect(int(x), int(y), 10, 5)
    
    def update(self, dt):
        self.pos_x += self.speed * self.direction * dt
        self.rect.x = int(self.pos_x)
        
    def draw(self, screen):
        pyg.draw.rect(screen, (255, 255, 0), self.rect)

class MainCharacter(Sprite):
    """Inheritance dari Sprite (Karakter Utama)"""
    def __init__(self, x, y, speed=400.0, hp=100.0):
        super().__init__(x, y, hp, speed)
        self.vel_y = 0
        self.is_grounded = False
        self.ammo = 3
        self.bullets = []
    
    def update(self, dt, width, height):
        keys = pyg.key.get_pressed()
        move_x = 0
        if keys[pyg.K_a]: move_x -= 1
        if keys[pyg.K_d]: move_x += 1
        
        # Jump
        if keys[pyg.K_w] and self.is_grounded:
            self.vel_y = -800 
            self.is_grounded = False

        # Apply Gravity
        self.vel_y += GRAVITY * dt
        self.pos_y += self.vel_y * dt
        self.pos_x += move_x * self.speed * dt
        
        self.rect.x = int(self.pos_x)
        self.rect.y = int(self.pos_y)
        
        # Grounding
        if self.rect.bottom > GROUND_Y:
            self.rect.bottom = GROUND_Y
            self.pos_y = float(self.rect.y)
            self.vel_y = 0
            self.is_grounded = True

        # Batasan Layar (Clamp)
        if self.rect.left < 0: self.pos_x = 0; self.rect.left = 0
        if self.rect.right > width: self.pos_x = width - self.rect.width; self.rect.right = width

        # Update Bullets
        for b in self.bullets[:]:
            b.update(dt)
            if b.rect.x < 0 or b.rect.x > width:
                self.bullets.remove(b)

    def shoot(self, target_pos):
        if self.ammo > 0:
            direction = 1 if target_pos[0] > self.rect.centerx else -1
            self.bullets.append(Peluru(self.rect.centerx, self.rect.centery, direction))
            self.ammo -= 1

    def draw(self, screen):
        pyg.draw.rect(screen, (0, 150, 255), self.rect)
        for b in self.bullets:
            b.draw(screen)

class Boss(Sprite):
    """Inheritance dari Sprite (Musuh / Frost Guardian)"""
    def __init__(self, x, speed=150.0, hp=100.0):
        # Boss position Y set based on GROUND_Y later
        super().__init__(x, 0, hp, speed)
        self.scale = 3.0
        
        # Load animations
        base_path = "Frost_Guardian_FREE_v1.0/PNG files"
        self.animations = {
            "idle": Animation(load_animation_frames(os.path.join(base_path, "idle")), 0.15),
            "walk": Animation(load_animation_frames(os.path.join(base_path, "walk")), 0.15),
            "attack": Animation(load_animation_frames(os.path.join(base_path, "1_atk")), 0.1, loop=False),
            "death": Animation(load_animation_frames(os.path.join(base_path, "death")), 0.1, loop=False),
            "hit": Animation(load_animation_frames(os.path.join(base_path, "take_hit")), 0.1, loop=False)
        }
        self.current_state = "SPAWN_IDLE"
        self.spawn_timer = 3.0
        
        self.width = int(192 * self.scale)
        self.height = int(128 * self.scale)
        self.rect = pyg.Rect(int(x), GROUND_Y - self.height, self.width, self.height)
        self.pos_y = float(self.rect.y)
        
        self.flip = False
        self.attack_hit_processed = False
        
    def update(self, dt, player_rect):
        anim_key = "idle"
        damage_to_player = 0
        
        if self.current_state == "SPAWN_IDLE":
            self.spawn_timer -= dt
            anim_key = "idle"
            if self.spawn_timer <= 0:
                self.current_state = "WALKING"
        
        elif self.current_state == "WALKING":
            anim_key = "walk"
            dist = player_rect.centerx - self.rect.centerx
            self.flip = dist > 0 # Face the player
            
            if abs(dist) > 150:
                direction = 1 if dist > 0 else -1
                self.pos_x += direction * self.speed * dt
            else:
                self.current_state = "ATTACKING"
                self.animations["attack"].reset()
                self.attack_hit_processed = False

        elif self.current_state == "ATTACKING":
            anim_key = "attack"
            anim = self.animations["attack"]
            
            if not self.attack_hit_processed and anim.current_frame >= 3:
                hit_rect = pyg.Rect(0, 0, 250, 250)
                if self.flip:
                    hit_rect.topleft = (self.rect.centerx, self.rect.y + 100)
                else:
                    hit_rect.topright = (self.rect.centerx, self.rect.y + 100)
                
                if hit_rect.colliderect(player_rect):
                    damage_to_player = 15
                    self.attack_hit_processed = True

            if anim.finished:
                self.current_state = "WALKING"
        
        self.rect.x = int(self.pos_x)
        self.animations[anim_key].update(dt)
        return damage_to_player

    def draw(self, screen):
        state_map = {
            "SPAWN_IDLE": "idle",
            "WALKING": "walk",
            "ATTACKING": "attack"
        }
        anim_key = state_map.get(self.current_state, "idle")
        frame = self.animations[anim_key].get_frame()
        if frame:
            frame = pyg.transform.scale(frame, (self.width, self.height))
            if self.flip:
                frame = pyg.transform.flip(frame, True, False)
            screen.blit(frame, self.rect)

# ==========================================
# 3. INHERITANCE: LAYAR MENU (MODULAR)
# ==========================================

class MenuLayar(Layar):
    """Inheritance dari Layar (Bisa jadi Main Menu, Game Over, atau Win)"""
    def __init__(self, width, height):
        super().__init__(width, height)
    
    def render_menu(self, screen, font_l, font_m, font_s):
        screen.fill((20, 20, 25))
        self.draw_text(screen, "DUEL SURVIVAL", font_l, (255, 255, 255), self.width//2, self.height//2 - 50)
        self.draw_text(screen, "WASD: Gerak | CLASH: Pose Tangan", font_s, (150, 150, 150), self.width//2, self.height//2 + 20)
        self.draw_text(screen, "Tekan [SPACE] untuk Mulai", font_m, (0, 255, 0), self.width//2, self.height//2 + 80)

    def render_end(self, screen, result_text, font_l, font_m, font_s):
        screen.fill((15, 15, 20))
        self.draw_text(screen, "THE END", font_l, (255, 0, 0), self.width//2, self.height//2 - 80)
        self.draw_text(screen, result_text, font_m, (255, 255, 255), self.width//2, self.height//2)
        self.draw_text(screen, "Tekan [R] untuk ke Menu Utama", font_s, (150, 150, 150), self.width//2, self.height//2 + 80)

# ==========================================
# 4. CLASS GAME MANAGER
# ==========================================

class GameApp:
    def __init__(self):
        pyg.init()
        self.width, self.height = 1280, 720
        self.screen = pyg.display.set_mode((self.width, self.height))
        pyg.display.set_caption("Duel Hand Pose: Survival (OOP Inheritance)")
        self.clock = pyg.time.Clock()
        
        # Fonts
        self.font_l = pyg.font.SysFont(None, 72)
        self.font_m = pyg.font.SysFont(None, 48)
        self.font_s = pyg.font.SysFont(None, 36)

        # Pose Detection
        self.labels = ["0", "1", "2", "3", "4", "5"]
        self.pose_app = ClientPose(noKamera=0, labels=self.labels, model_path="model.h5", conf_min=0.6)
        if not self.pose_app.buka_camera():
            print("Kamera gagal dibuka!")
            sys.exit()

        # UI Manager
        self.ui_manager = MenuLayar(self.width, self.height)
        
        # State
        self.state = 'MENU'
        self.running = True
        self.result_text = ""
        
        # Sounds
        pyg.mixer.init()
        self.sfx_ushirabi = pyg.mixer.Sound("Audio n Sound Effect/USHIRABI.mp3")
        self.sfx_granit = pyg.mixer.Sound("Audio n Sound Effect/Granit Blash.mp3")
        
        # Entities
        self.player = MainCharacter(self.width//4, GROUND_Y - 40)
        self.enemy = None
        self.clash_timer = 0
        self.clash_countdown = 0
        self.enemy_choice = -1
        self.clash_player_label = -1
        self.clash_enemy_label = -1
        self.clash_sound_played = False

    def reset_battle(self):
        self.player = MainCharacter(self.width//4, GROUND_Y - 40)
        self.enemy = Boss(self.width - 250)
        self.clash_timer = float(random.randint(5, 15))
        try: cv2.destroyAllWindows()
        except: pass

    def draw_hud(self):
        # Background bar
        pyg.draw.rect(self.screen, (50, 50, 50), (20, 20, 200, 25))
        pyg.draw.rect(self.screen, (50, 50, 50), (self.width - 220, 20, 200, 25))
        
        # HP Bars
        p_width = int(200 * (max(0, self.player.hp) / 100.0))
        pyg.draw.rect(self.screen, (0, 255, 100), (20, 20, p_width, 25))
        e_width = int(200 * (max(0, self.enemy.hp) / 100.0))
        pyg.draw.rect(self.screen, (255, 50, 50), (self.width - 220, 20, e_width, 25))
        
        self.ui_manager.draw_text(self.screen, f"PLAYER: {int(self.player.hp)}", self.font_s, (255, 255, 255), 25, 25, False)
        self.ui_manager.draw_text(self.screen, f"FROST: {int(self.enemy.hp)}", self.font_s, (255, 255, 255), self.width - 215, 25, False)

    def handle_events(self):
        for event in pyg.event.get():
            if event.type == pyg.QUIT:
                self.running = False
            if event.type == pyg.MOUSEBUTTONDOWN:
                if self.state == 'PLAYING' and event.button == 1:
                    self.player.shoot(event.pos)
            if event.type == pyg.KEYDOWN:
                if self.state == 'MENU' and event.key == pyg.K_SPACE:
                    self.reset_battle()
                    self.state = 'PLAYING'
                elif self.state == 'END' and event.key == pyg.K_r:
                    self.state = 'MENU'
                elif event.key == pyg.K_ESCAPE:
                    self.running = False

    def update_clash(self, dt, fitur):
        if self.clash_countdown > 0:
            self.clash_countdown -= dt
            if self.clash_countdown <= 0:
                # Kunci label pas detik 0
                self.clash_player_label = int(fitur["Rl"])
                self.clash_enemy_label = self.enemy_choice
                
                print(f"[DEBUG] Clash Triggered! Player: {self.clash_player_label} vs Boss: {self.clash_enemy_label}")
                
                # Trigger Sound (Paralel)
                # Sekarang Label 1 juga pake OR (salah satu pilih 1, bunyi)
                if self.clash_player_label == 1 or self.clash_enemy_label == 1:
                    self.sfx_ushirabi.play()
                
                if self.clash_player_label == 3 or self.clash_enemy_label == 3:
                    self.sfx_granit.play()
                
                self.clash_sound_played = True
                # Kasih waktu dikit buat mixer mulai muter
                pyg.time.delay(100) 
            return

        # Tunggu SFX Selesai (Freeze Logic)
        # Kita cek apakah channel yang muter sound ini masih aktif
        if self.sfx_ushirabi.get_num_channels() > 0 or self.sfx_granit.get_num_channels() > 0:
            return

        # Setelah Audio Beres, Baru Proses Damage
        player_label = self.clash_player_label
        enemy_label = self.clash_enemy_label
        
        damage_p = 0
        damage_e = 0
        
        if player_label == -1:
            # Player tidak terdeteksi pose-nya
            damage_p = 15 
            print("[DEBUG] Player pose not detected! Penalty applied.")
        elif player_label == enemy_label:
            # Draw / Seri
            print(f"[DEBUG] Draw! Both chose {player_label}. No damage.")
        else:
            # Logic Clash Utama
            # 0 > 4, 5 > 0, 1 > 2, 2 > 3, 3 > 1
            if (player_label == 0 and enemy_label == 4) or \
               (player_label == 5 and enemy_label == 0) or \
               (player_label == 1 and enemy_label == 2) or \
               (player_label == 2 and enemy_label == 3) or \
               (player_label == 3 and enemy_label == 1) or \
               (player_label == 4 and enemy_label in [1, 2, 3]):
                damage_e = 25
                print(f"[DEBUG] Player Wins Clash! Damage to Boss: {damage_e}")
            else:
                # Kebalikannya, Boss yang menang
                damage_p = 20
                print(f"[DEBUG] Boss Wins Clash! Damage to Player: {damage_p}")
        
        # Eksekusi pengurangan darah
        self.player.hp -= damage_p
        self.enemy.hp -= damage_e
        
        self.clash_sound_played = False 

        # Cek Kondisi Menang/Kalah
        if self.player.hp <= 0:
            self.state = 'END'
            self.result_text = f"KALAH ADU! (Anda:{player_label}, Musuh:{enemy_label})"
        elif self.enemy.hp <= 0:
            self.state = 'END'
            self.result_text = f"MENANG ADU! (Anda:{player_label}, Musuh:{enemy_label})"
        else:
            self.state = 'PLAYING'
            self.clash_timer = float(random.randint(5, 15))
            try: cv2.destroyWindow("Adu Pose!")
            except: pass

    def run(self):
        while self.running:
            dt = self.clock.tick(120) / 1000.0
            self.handle_events()
            
            if self.state == 'MENU':
                self.pose_app.cap.grab()
                self.ui_manager.render_menu(self.screen, self.font_l, self.font_m, self.font_s)
                
            elif self.state == 'PLAYING':
                self.pose_app.cap.grab()
                self.player.update(dt, self.width, self.height)
                
                if self.enemy:
                    damage = self.enemy.update(dt, self.player.rect)
                    if damage > 0:
                        self.player.hp -= damage
                        if self.player.hp <= 0:
                            self.state = 'END'
                            self.result_text = "DIKALAHKAN OLEH FROST GUARDIAN!"
                    
                    for b in self.player.bullets[:]:
                        if b.rect.colliderect(self.enemy.rect):
                            self.enemy.hp -= b.damage
                            self.player.bullets.remove(b)
                            if self.enemy.hp <= 0:
                                self.state = 'END'
                                self.result_text = "MENANG! FROST GUARDIAN DIKALAHKAN!"

                self.clash_timer -= dt
                if self.clash_timer <= 0:
                    self.state = 'CLASH'
                    self.clash_countdown = 5.0
                    self.enemy_choice = random.randint(0, 5)
                    self.player.ammo = 3

                # Render Playing
                self.screen.fill((20, 20, 25))
                self.draw_hud()
                self.player.draw(self.screen)
                if self.enemy: self.enemy.draw(self.screen)
                self.ui_manager.draw_text(self.screen, f"AMMO: {self.player.ammo}", self.font_s, (255, 255, 0), 110, 80)
                self.ui_manager.draw_text(self.screen, f"Next Clash: {max(0.0, self.clash_timer):.1f}s", self.font_s, (255, 255, 0), 110, 50)
                
            elif self.state == 'CLASH':
                fitur = {"Rl": -1}
                ret, frame = self.pose_app.cap.read()
                if ret:
                    frame, fitur = self.pose_app.process_frame(frame)
                    cv2.imshow("Adu Pose!", frame)
                    cv2.waitKey(1)
                
                self.update_clash(dt, fitur)
                
                # Render Clash Screen
                self.screen.fill((60, 30, 0))
                self.draw_hud()
                self.ui_manager.draw_text(self.screen, "ADU KEKUATAN!", self.font_l, (255, 200, 0), self.width//2, 100)
                self.ui_manager.draw_text(self.screen, f"SIAPKAN POSE: {max(0.0, self.clash_countdown):.1f}", self.font_m, (255, 255, 255), self.width//2, self.height//2)
                self.ui_manager.draw_text(self.screen, f"Terdeteksi: {fitur['Rl']}", self.font_s, (0, 255, 255), self.width//2, self.height//2 + 60)

            elif self.state == 'END':
                self.pose_app.cap.grab()
                self.ui_manager.render_end(self.screen, self.result_text, self.font_l, self.font_m, self.font_s)
                
            pyg.display.flip()

        self.pose_app.close()
        pyg.quit()

if __name__ == "__main__":
    app = GameApp()
    app.run()
