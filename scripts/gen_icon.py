"""Generates assets/icon.rgba (64x64 raw RGBA, 8-bit) for the window titlebar
and taskbar: the navy rounded-square tile with the monochrome shredding glyph
(sheet, slit, falling strips). Drawn at 8x and downsampled so the rounded
corners get real alpha, which ImageMagick's SVG delegate loses."""
from PIL import Image, ImageDraw

SCALE = 8
SIZE = 64 * SCALE
INK = (234, 242, 247, 255)      # theme off-white #EAF2F7
TILE = (11, 18, 32, 255)        # canvas navy #0B1220
FOLD = (234, 242, 247, 140)     # sheet fold at 55% opacity

img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

d.rounded_rectangle((0, 0, SIZE - 1, SIZE - 1), radius=14 * SCALE, fill=TILE)

# sheet with folded corner (12,5)-(40,25)
d.polygon([(12*SCALE, 5*SCALE), (32*SCALE, 5*SCALE), (40*SCALE, 13*SCALE),
           (40*SCALE, 25*SCALE), (12*SCALE, 25*SCALE)], fill=INK)
d.polygon([(32*SCALE, 5*SCALE), (32*SCALE, 13*SCALE), (40*SCALE, 13*SCALE)], fill=FOLD)

# shredder slit
d.rounded_rectangle((12*SCALE, 29*SCALE, 40*SCALE, 33.5*SCALE), radius=1.75*SCALE, fill=INK)

# falling strips
for x, h in [(14, 10), (19.3, 15), (24.6, 8), (29.9, 16), (35.2, 9)]:
    d.rounded_rectangle((x*SCALE, 36.5*SCALE, (x + 3.2)*SCALE, (36.5 + h)*SCALE),
                        radius=1.6*SCALE, fill=INK)

img = img.resize((64, 64), Image.LANCZOS)

# Pillow has no raw-RGBA writer; write the bytes directly.
raw = img.tobytes()
with open(r"C:\Repository\GitHub\PaperRL\assets\icon.rgba", "wb") as f:
    f.write(raw)
print("wrote", len(raw), "bytes; min alpha:", min(raw[3::4]))
