###### IMPORT ######
import os
import sys

import numpy as np
import pygame
from PIL import Image

###### CONSTANTS ######
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')
THUMB = 256

WIDTH, HEIGHT = 640, 640
TOP_HEIGHT = 320
BG = (30, 30, 34)
FG = (230, 230, 230)
MUTED = (140, 140, 150)
TRACK = (70, 70, 80)

###### COLOR MATH ######
# All functions work on plain ints/floats or on numpy arrays (elementwise),
# so the same code drives both the sliders and the whole-image transform.
def luminance(r, g, b):
    return (r + g + b) / 3

def chroma(r, g, b):
    l = luminance(r, g, b)
    return np.maximum(np.maximum(np.abs(r - l), np.abs(g - l)), np.abs(b - l))

def transform(r, g, b, l, c):
    maxChannel = np.maximum(np.maximum(r, g), b)
    whiteWeight = (maxChannel / 255) ** 8
    
    newR = r * (1 - whiteWeight) + 255 * (whiteWeight)
    newG = g * (1 - whiteWeight) + 255 * (whiteWeight)
    newB = b * (1 - whiteWeight) + 255 * (whiteWeight)
    return newR, newG, newB

def applyTransform(r, g, b):
    l = luminance(r, g, b)
    c = chroma(r, g, b)
    return transform(r, g, b, l, c)

###### IMAGES ######
def listImages():
    return sorted(f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(IMAGE_EXTS))

def loadSquare(path):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize((THUMB, THUMB), Image.LANCZOS)
    return np.array(img).astype(np.float32)

def arrayToSurface(arr):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    # pygame surfarray is (width, height, 3), numpy image is (height, width, 3)
    return pygame.surfarray.make_surface(arr.transpose(1, 0, 2))

def buildSurfaces(path):
    arr = loadSquare(path)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    newR, newG, newB = applyTransform(r, g, b)
    newArr = np.stack(np.broadcast_arrays(newR, newG, newB), axis=-1)
    return arrayToSurface(arr), arrayToSurface(newArr)

###### UI ######
class Slider:
    def __init__(self, label, color, x, y, w, value=128):
        self.label = label
        self.color = color
        self.rect = pygame.Rect(x, y, w, 8)
        self.value = value
        self.dragging = False

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.inflate(0, 24).collidepoint(event.pos):
                self.dragging = True
                self.setFromX(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.setFromX(event.pos[0])

    def setFromX(self, x):
        t = (x - self.rect.x) / self.rect.w
        self.value = int(round(max(0, min(1, t)) * 255))

    def draw(self, screen, font):
        screen.blit(font.render(f'{self.label} = {self.value}', True, FG), (self.rect.x, self.rect.y - 26))
        pygame.draw.rect(screen, TRACK, self.rect, border_radius=4)
        filled = self.rect.copy()
        filled.w = int(self.rect.w * self.value / 255)
        pygame.draw.rect(screen, self.color, filled, border_radius=4)
        knobX = self.rect.x + filled.w
        pygame.draw.circle(screen, FG, (knobX, self.rect.centery), 9)

def drawSwatch(screen, font, label, color, x, y, size=56):
    pygame.draw.rect(screen, color, (x, y, size, size), border_radius=6)
    pygame.draw.rect(screen, MUTED, (x, y, size, size), 1, border_radius=6)
    screen.blit(font.render(label, True, MUTED), (x, y + size + 4))

###### MAIN ######
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Color Transform')
    font = pygame.font.SysFont('consolas', 18)
    small = pygame.font.SysFont('consolas', 14)
    clock = pygame.time.Clock()

    sliders = [
        Slider('r', (220, 70, 70), 30, 50, 260, 176),
        Slider('g', (70, 200, 90), 30, 120, 260, 199),
        Slider('b', (70, 110, 230), 30, 190, 260, 198),
    ]

    images = listImages()
    if not images:
        print(f'No images found in {IMAGE_DIR}')
        sys.exit(1)
    index = 0
    original, transformed = buildSurfaces(os.path.join(IMAGE_DIR, images[index]))

    pad = (WIDTH - 2 * THUMB) // 3
    leftRect = pygame.Rect(pad, TOP_HEIGHT + 30, THUMB, THUMB)
    rightRect = pygame.Rect(2 * pad + THUMB, TOP_HEIGHT + 30, THUMB, THUMB)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            for s in sliders:
                s.handle(event)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if leftRect.collidepoint(event.pos) or rightRect.collidepoint(event.pos):
                    index = (index + 1) % len(images)
                    original, transformed = buildSurfaces(os.path.join(IMAGE_DIR, images[index]))

        r, g, b = (s.value for s in sliders)
        l = luminance(r, g, b)
        c = chroma(r, g, b)
        newR, newG, newB = transform(r, g, b, l, c)
        newColor = tuple(int(np.clip(round(float(v)), 0, 255)) for v in (newR, newG, newB))
        gray = int(round(l))

        screen.fill(BG)

        # --- top section ---
        for s in sliders:
            s.draw(screen, font)

        infoX = 340
        screen.blit(font.render(f'l = {l:.1f}', True, FG), (infoX, 24))
        screen.blit(font.render(f'c = {float(c):.1f}', True, FG), (infoX, 50))
        screen.blit(font.render(f"r' = {newColor[0]}", True, FG), (infoX, 90))
        screen.blit(font.render(f"g' = {newColor[1]}", True, FG), (infoX, 114))
        screen.blit(font.render(f"b' = {newColor[2]}", True, FG), (infoX, 138))

        drawSwatch(screen, small, 'input', (r, g, b), infoX, 190)
        drawSwatch(screen, small, 'luminance', (gray, gray, gray), infoX + 76, 190)
        drawSwatch(screen, small, 'output', newColor, infoX + 152, 190)

        pygame.draw.line(screen, TRACK, (0, TOP_HEIGHT), (WIDTH, TOP_HEIGHT), 1)

        # --- bottom section ---
        screen.blit(original, leftRect)
        screen.blit(transformed, rightRect)
        screen.blit(small.render('original', True, MUTED), (leftRect.x, leftRect.bottom + 6))
        screen.blit(small.render('transformed', True, MUTED), (rightRect.x, rightRect.bottom + 6))
        name = small.render(f'{images[index]}  (click to cycle)', True, MUTED)
        screen.blit(name, ((WIDTH - name.get_width()) // 2, TOP_HEIGHT + 8))

        pygame.display.flip()
        clock.tick(60)

if __name__ == '__main__':
    main()
