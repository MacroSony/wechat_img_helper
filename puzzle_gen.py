import os
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class ImgPuzzle:
    background_color: tuple[int, int, int]
    width: int
    height: int
    row_height: int
    row_spacing: int
    imgs: list

    def to_img(self) -> np.ndarray:
        puzzle_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        puzzle_img[:] = self.background_color
        current_y = 0
        image_index = 0

        while image_index < len(self.imgs) and current_y + self.row_height <= self.height:
            current_row_imgs = []
            current_row_widths = []
            sum_width = 0

            # Collect images that fit into the current row
            for i in range(image_index, len(self.imgs)):
                img = self.imgs[i]
                original_h, original_w = img.shape[:2]
                if original_h == 0:
                    new_width = 0
                else:
                    new_width = int(original_w * (self.row_height / original_h))
                if sum_width + new_width > self.width:
                    break
                current_row_imgs.append(img)
                current_row_widths.append(new_width)
                sum_width += new_width

            n = len(current_row_imgs)
            if n == 0:
                # Skip the image that cannot fit even alone
                image_index += 1
                continue

            remaining_space = self.width - sum_width
            x_positions = []

            if n == 1:
                # Center the single image
                x = (self.width - current_row_widths[0]) // 2
                x_positions.append(x)
            else:
                # Calculate spacing between images
                spacing = remaining_space / (n - 1)
                current_x = 0.0
                for w in current_row_widths:
                    x_positions.append(int(round(current_x)))
                    current_x += w + spacing

            # Adjust positions and sizes to prevent overflow
            for i in range(n):
                x = x_positions[i]
                w = current_row_widths[i]
                if x + w > self.width:
                    current_row_widths[i] = self.width - x

            # Resize and place images
            for i in range(n):
                img = current_row_imgs[i]
                original_h, original_w = img.shape[:2]
                if original_h == 0:
                    continue  # Skip invalid image
                new_width = current_row_widths[i]
                new_height = self.row_height
                resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                x = x_positions[i]
                y = current_y

                # Ensure the resized image does not exceed puzzle boundaries
                if y + new_height > self.height:
                    new_height = self.height - y
                    resized_img = resized_img[:new_height, :]
                if x + new_width > self.width:
                    new_width = self.width - x
                    resized_img = resized_img[:, :new_width]

                puzzle_img[y:y+new_height, x:x+new_width] = resized_img

            image_index += n
            current_y += self.row_height + self.row_spacing

        return puzzle_img

def construct_puzzle(images_path, background_color=(0, 0, 0), puzzle_width=3240, puzzle_height=3240, row_height=512, row_spacing=40) -> ImgPuzzle:
    puzzle = ImgPuzzle(background_color=background_color, width=puzzle_width, 
                        height=puzzle_height, row_height=row_height, 
                        row_spacing=row_spacing, imgs=[])

    imgs = []

    for image_name in os.listdir(images_path):
        image_path = os.path.join(images_path, image_name)
        img = cv2.imread(image_path)
        imgs.append(img)
    
    puzzle.imgs = imgs

    return puzzle

# --- Example Usage ---
if __name__ == '__main__':
    image_folder = "selected_imgs"  # Replace with the path to your image folder

    puzzle = construct_puzzle(image_folder)

    cv2.imwrite("textimg.png", puzzle.to_img())