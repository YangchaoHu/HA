from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SRC = Path(r"C:\Users\dell\Desktop\soft_ware_code\optimize\PNGFILE")
OUT = Path(r"C:\Users\dell\Desktop\HA\HA\output\poster_ha\engineering_contact_sheet.png")
FONT = r"C:\Windows\Fonts\msyh.ttc"

imgs = []
for p in SRC.glob("*.png"):
    try:
        with Image.open(p) as im:
            w, h = im.size
        if p.stat().st_size > 100_000:
            imgs.append((p.stat().st_size, p, w, h))
    except Exception:
        pass

imgs = sorted(imgs, reverse=True)[:36]
thumb_w, thumb_h = 240, 170
cols = 4
rows = (len(imgs) + cols - 1) // cols
sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 42)), "white")
draw = ImageDraw.Draw(sheet)
font = ImageFont.truetype(FONT, 14)

for idx, (_, p, w, h) in enumerate(imgs):
    r, c = divmod(idx, cols)
    x, y = c * thumb_w, r * (thumb_h + 42)
    im = Image.open(p).convert("RGB")
    im.thumbnail((thumb_w - 10, thumb_h - 10), Image.Resampling.LANCZOS)
    sheet.paste(im, (x + (thumb_w - im.width) // 2, y + 5))
    draw.text((x + 8, y + thumb_h + 6), f"{idx+1}. {p.name}", font=font, fill=(0, 0, 0))
    draw.text((x + 8, y + thumb_h + 24), f"{w}x{h}", font=font, fill=(80, 80, 80))

sheet.save(OUT)
print(OUT)
