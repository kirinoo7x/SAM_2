"""
Generate synthetic crack images for testing the SAM2 crack detection model.
"""
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os

# Create output directory
output_dir = "test_images"
os.makedirs(output_dir, exist_ok=True)

def create_linear_crack(width=512, height=512):
    """Create an image with a simple linear crack."""
    # Create background (concrete texture)
    img = np.random.randint(180, 220, (height, width, 3), dtype=np.uint8)

    # Add some texture
    noise = np.random.normal(0, 10, (height, width, 3))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # Draw a diagonal crack
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Main crack line with some variation
    points = []
    for i in range(0, width, 10):
        y = int(height * i / width + np.random.randint(-20, 20))
        points.append((i, y))

    draw.line(points, fill=(50, 50, 50), width=3)

    # Add smaller crack branches
    for x, y in points[::5]:
        branch_points = [(x, y)]
        for j in range(1, 4):
            bx = x + j * 15 + np.random.randint(-5, 5)
            by = y + j * 15 + np.random.randint(-5, 5)
            branch_points.append((bx, by))
        draw.line(branch_points, fill=(60, 60, 60), width=2)

    return pil_img

def create_branching_crack(width=512, height=512):
    """Create an image with branching crack pattern."""
    # Create background
    img = np.random.randint(170, 210, (height, width, 3), dtype=np.uint8)
    noise = np.random.normal(0, 15, (height, width, 3))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Start from center and branch out
    start_x, start_y = width // 2, height // 4

    def draw_branch(x, y, angle, length, depth=0):
        if depth > 3 or length < 20:
            return

        # Calculate end point
        end_x = int(x + length * np.cos(angle))
        end_y = int(y + length * np.sin(angle))

        # Add some randomness to the line
        points = [(x, y)]
        steps = max(int(length / 20), 2)
        for i in range(1, steps):
            t = i / steps
            px = int(x + (end_x - x) * t + np.random.randint(-5, 5))
            py = int(y + (end_y - y) * t + np.random.randint(-5, 5))
            points.append((px, py))
        points.append((end_x, end_y))

        width_line = max(4 - depth, 1)
        draw.line(points, fill=(40, 40, 40), width=width_line)

        # Branch out
        if np.random.random() > 0.3:
            draw_branch(end_x, end_y, angle - np.pi/6 + np.random.uniform(-0.2, 0.2),
                       length * 0.7, depth + 1)
        if np.random.random() > 0.3:
            draw_branch(end_x, end_y, angle + np.pi/6 + np.random.uniform(-0.2, 0.2),
                       length * 0.7, depth + 1)
        if np.random.random() > 0.5:
            draw_branch(end_x, end_y, angle + np.random.uniform(-0.3, 0.3),
                       length * 0.8, depth + 1)

    # Draw main branches
    draw_branch(start_x, start_y, np.pi/2, 150)
    draw_branch(start_x, start_y, np.pi/2 - np.pi/4, 120)
    draw_branch(start_x, start_y, np.pi/2 + np.pi/4, 120)

    return pil_img

def create_horizontal_crack(width=512, height=512):
    """Create an image with horizontal crack."""
    # Create background
    img = np.random.randint(175, 215, (height, width, 3), dtype=np.uint8)
    noise = np.random.normal(0, 12, (height, width, 3))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Draw horizontal crack with waves
    y_center = height // 2
    points = []
    for x in range(0, width, 8):
        y = y_center + int(15 * np.sin(x * 0.02)) + np.random.randint(-10, 10)
        points.append((x, y))

    draw.line(points, fill=(45, 45, 45), width=4)

    # Add some secondary cracks
    for i in range(3):
        y_offset = y_center + np.random.randint(-100, 100)
        sec_points = []
        for x in range(0, width, 15):
            y = y_offset + int(8 * np.sin(x * 0.03)) + np.random.randint(-8, 8)
            sec_points.append((x, y))
        draw.line(sec_points, fill=(55, 55, 55), width=2)

    return pil_img

def create_multiple_cracks(width=512, height=512):
    """Create an image with multiple small cracks."""
    # Create background
    img = np.random.randint(185, 225, (height, width, 3), dtype=np.uint8)
    noise = np.random.normal(0, 8, (height, width, 3))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Draw multiple random cracks
    for _ in range(8):
        start_x = np.random.randint(50, width - 50)
        start_y = np.random.randint(50, height - 50)
        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.randint(50, 150)

        points = [(start_x, start_y)]
        for i in range(1, 10):
            t = i / 10
            x = int(start_x + length * t * np.cos(angle) + np.random.randint(-5, 5))
            y = int(start_y + length * t * np.sin(angle) + np.random.randint(-5, 5))
            points.append((x, y))

        draw.line(points, fill=(50, 50, 50), width=np.random.randint(2, 4))

    return pil_img

def create_clean_surface(width=512, height=512):
    """Create an image without cracks (control image)."""
    # Create background
    img = np.random.randint(190, 230, (height, width, 3), dtype=np.uint8)
    noise = np.random.normal(0, 5, (height, width, 3))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # Add some subtle texture variation
    for _ in range(20):
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        radius = np.random.randint(5, 15)
        color_offset = np.random.randint(-10, 10)
        cv2.circle(img, (x, y), radius,
                  (int(200 + color_offset), int(200 + color_offset), int(200 + color_offset)),
                  -1)

    # Blur to make it look more natural
    img = cv2.GaussianBlur(img, (5, 5), 0)

    return Image.fromarray(img)

# Generate all test images
print("Generating test images...")

images = {
    "linear_crack.jpg": create_linear_crack(),
    "branching_crack.jpg": create_branching_crack(),
    "horizontal_crack.jpg": create_horizontal_crack(),
    "multiple_cracks.jpg": create_multiple_cracks(),
    "clean_surface.jpg": create_clean_surface(),
}

for filename, img in images.items():
    filepath = os.path.join(output_dir, filename)
    img.save(filepath)
    print(f"Created: {filepath}")

print(f"\nAll test images saved to '{output_dir}/' directory")
print("\nGenerated images:")
print("1. linear_crack.jpg - Simple diagonal crack")
print("2. branching_crack.jpg - Complex branching pattern")
print("3. horizontal_crack.jpg - Horizontal wavy crack")
print("4. multiple_cracks.jpg - Multiple small cracks")
print("5. clean_surface.jpg - No cracks (control)")
