# import os
# import cv2
# import numpy as np


# class PuzzleGenerator:
#     def __init__(self, images_path):
#         self.path = images_path

#     def generate_puzzle(self):
#         puzzle = np.zeros((3240,3240,3))
#         imgs = []
#         for image in os.listdir(self.path):
#             img = cv2.imread(os.path.join(self.path, image))
#             scale_ratio = 520 / img.shape[0]
#             height = int(img.shape[1] * scale_ratio)
#             img_small = cv2.resize(img, (height, 520), interpolation=cv2.INTER_AREA)
#             imgs.append(img_small)
        
#         total_width = 0
#         for img in imgs:
#             w, _, _ = img.shape
#             total_width += w

#         print(total_width)
#         # cv2.imwrite("test_blank.png", puzzle)

# pg = PuzzleGenerator("selected_imgs")

# pg.generate_puzzle()

import os
import cv2
import numpy as np


class PuzzleGenerator:
    def __init__(self, images_path):
        self.path = images_path

    def generate_puzzle(self, puzzle_width=3240, puzzle_height=3240, row_height=480, row_spacing=40):
        """
        Generates a puzzle image from a directory of images and crops it into 9 1:1 images.

        Args:
            puzzle_width: The desired width of the puzzle image.
            puzzle_height: The desired height of the puzzle image.
            row_height: The desired height of each row of images.
            row_spacing: The fixed spacing between rows (in pixels).

        Returns:
            A list of 9 cropped 1:1 images as NumPy arrays, or None if there was an error.
        """

        puzzle = np.zeros((puzzle_height, puzzle_width, 3), dtype=np.uint8)
        imgs = []
        ignored_images = []

        # 1. Read and Resize Images (Maintaining Aspect Ratio)
        for image_name in os.listdir(self.path):
            image_path = os.path.join(self.path, image_name)
            img = cv2.imread(image_path)

            if img is None:  # Handle potential read errors
                print(f"Error: Could not read image {image_name}. Skipping.")
                ignored_images.append(image_name)
                continue

            # Calculate scaling based on desired row_height
            scale_ratio = row_height / img.shape[0]
            new_width = int(img.shape[1] * scale_ratio)
            new_height = int(img.shape[0] * scale_ratio)

            resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            imgs.append((resized_img, image_name))  # Store image and original name


        # 2. Arrange Images into Rows
        current_row_y = 0  # Track the current row's y-coordinate
        current_row_images = []
        current_row_width = 0
        rows = []

        for img, image_name in imgs:
            if current_row_width + img.shape[1] <= puzzle_width:
                # Image fits in the current row
                current_row_images.append((img, image_name))
                current_row_width += img.shape[1]
            else:
                # Image doesn't fit; start a new row
                rows.append((current_row_images, current_row_width))  # Store the completed row
                current_row_images = [(img, image_name)] # Start a new row with current img
                current_row_width = img.shape[1]

        # Add the last row if it's not empty
        if current_row_images:
            rows.append((current_row_images, current_row_width))


        # 3. Place Rows onto Puzzle Image
        for row_images, row_width in rows:
            if current_row_y + row_height > puzzle_height:
                 print(f"Warning: Not enough vertical space for all rows. Skipping remaining rows.")
                 break # Stop if not space for current row.

            # Calculate spacing for the current row
            num_images = len(row_images)
            if num_images > 1:
                spacing = (puzzle_width - row_width) // (num_images - 1)
            else:
                spacing = 0  # If only one image, no spacing needed (left-align)

            current_x = 0
            for img, image_name in row_images:
                # Place the image onto the puzzle
                puzzle[current_row_y:current_row_y + img.shape[0], current_x:current_x + img.shape[1]] = img
                current_x += img.shape[1] + spacing

            current_row_y += row_height + row_spacing #Increment position for next row.

        #4. Handle and Display ignored images.
        for img, image_name in imgs:
            used = False
            for row in rows:
                if any(img_name == image_name for _, img_name in row[0]):
                    used = True
                    break
            if not used:
                ignored_images.append(image_name)

        if ignored_images:
            print("Ignored images (due to size/fitting issues):")
            for image_name in ignored_images:
                print(f"- {image_name}")
        
        # --- Cropping into 9 (1:1) Images ---
        cropped_images = []
        crop_size = puzzle_width // 3  # Since puzzle is square (3240x3240), width/3 = height/3

        for i in range(3):
            for j in range(3):
                x_start = j * crop_size
                y_start = i * crop_size
                x_end = x_start + crop_size
                y_end = y_start + crop_size

                cropped_img = puzzle[y_start:y_end, x_start:x_end]
                cropped_images.append(cropped_img)

        return cropped_images

# --- Example Usage ---
if __name__ == '__main__':
    image_folder = "selected_imgs"  # Replace with the path to your image folder

     # Create the "images" folder if it doesn't exist (for testing)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        print(f"Created directory: {image_folder}. Please put some images in it to test.")
    else:
        generator = PuzzleGenerator(image_folder)
        cropped_puzzle_images = generator.generate_puzzle()

        if cropped_puzzle_images:
            # Create a directory to save cropped images if it doesn't exist.
            output_dir = "cropped_puzzles"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Display and save each cropped image
            for i, cropped_img in enumerate(cropped_puzzle_images):
                # cv2.imshow(f"Cropped Puzzle {i+1}", cropped_img)
                cv2.imwrite(os.path.join(output_dir, f"puzzle_part_{i+1}.png"), cropped_img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()