#Create the PNG for the Xiangqi game - circle with chinese character in the center
from PIL import Image, ImageDraw, ImageFont
import os

def create_piece(chinese_character, color):
    # Create image
    size = 400
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw circle
    if color=='RED':
        circle_color = (255, 0, 0)  # Red

    elif color=='BLACK':
        circle_color = (0, 0, 0)  # Black

    draw.ellipse([0, 0, size-1, size-1], fill=circle_color)

    # Draw Chinese character
    font_size = 200
    font = ImageFont.truetype("simhei.ttf", font_size)  # Use Chinese font
    text = chinese_character
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((size - text_width) // 2, (size - text_height) // 2 - 20)

    draw.text(position, text, fill=(255, 255, 255), font=font)

    # Save
    img.save(os.path.join(os.getcwd(), "PyXiangQi\\GUI\\pieces_canvas", chinese_character + '.png'))

xiangqi_pieces_red = ["帥","車","馬","炮","兵","士","相"]
xiangqi_pieces_black = ["將","俥","傌","砲","卒","仕","象"]

for piece in xiangqi_pieces_red:
        create_piece(piece, 'RED')

for piece in xiangqi_pieces_black:
        create_piece(piece, 'BLACK')

# Create image
size = 400
img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)
circle_color = (70, 70, 70,120)  # Red
draw.ellipse([0, 0, size-1, size-1], fill=circle_color)

# Save
img.save(os.path.join(os.getcwd(), "PyXiangQi\\GUI\\pieces_canvas",  'spot.png'))