import cv2 as cv
import numpy as np
import logging

def filter_contours_by_size_and_shape(contours, min_size=100, max_size=1000):
    """
    Filters contours by size and roundness.
    """
    filtered_contours = []
    for contour in contours:
        area = cv.contourArea(contour)
        if min_size < area < max_size:
            x, y, w, h = cv.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.8 <= aspect_ratio <= 1.2:  # Nearly round shapes
                filtered_contours.append(contour)
    return filtered_contours


def get_position(contour, cell_size_x, cell_size_y):
    """
    Calculates the position of the contour on the grid.
    """
    m = cv.moments(contour)
    if m["m00"] > 0:
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        grid_x = int(cx // cell_size_x)  # Column
        grid_y = int(cy // cell_size_y)  # Row
        return grid_x, grid_y, cx, cy
    return None

def get_cell_based_position(cX, cY, cell_size_x, cell_size_y):
    part_x = (cX % cell_size_x) / cell_size_x
    part_y = (cY % cell_size_y) / cell_size_y
    return part_x, part_y

def get_mask_red(hsv_image):
    # Define HSV color ranges for red and blue
    lower_red1 = np.array([0, 80, 30])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 80, 30])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv.inRange(hsv_image, lower_red2, upper_red2)
    mask_red = cv.bitwise_or(mask_red1, mask_red2)
    return mask_red

def get_mask_blue(hsv_image):
    lower_blue = np.array([80, 80, 30])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv.inRange(hsv_image, lower_blue, upper_blue)
    return mask_blue

class ImageDetector:
    """
    A class for detecting and analyzing positions on a chessboard-like grid.
    """

    def __init__(self, image, robot_is_first, padding=0.2):
        self.error = None
        self.original_image = image
        transformation_matrix, warped_image = self._preprocess_image(image)
        if transformation_matrix is not None and warped_image is not None:
            self.transformation_matrix = transformation_matrix
            self.image = warped_image
            self.padding = padding
            self.board_matrix = np.zeros((8, 8), dtype=np.uint8)
            self.positions = {"blue": [], "red": []}
            self.piece_positions = {}
            self._process_image()
            self._set_rotation()
            self.box = self.get_captured_box(image)
            self.new_king_position = self.calculate_new_king_position(robot_is_first)

    def get_error(self):
        return self.error

    def _preprocess_image(self, image_origin):
        """
        Preprocesses the image to detect a chessboard-like contour, applies a perspective transformation, and saves the calibrated image.
        :return: Transformation matrix for mapping coordinates, or None if no chessboard contour is found.
        """

        def order_points(pts):
            """
            Orders points in a consistent order: top-left, top-right, bottom-right, bottom-left.
            """
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]  # Top-left
            rect[2] = pts[np.argmax(s)]  # Bottom-right

            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]  # Top-right
            rect[3] = pts[np.argmax(diff)]  # Bottom-left
            return rect

        # Load image
        img = image_origin

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray_blurred = cv.GaussianBlur(gray, (5, 5), 0)

        # Detect edges and contours
        edges = cv.Canny(gray_blurred, 180, 220)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        edges_cleaned = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv.findContours(
            edges_cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        chessboard_contour = None

        # Find the largest contour
        max_area = 0
        max_index = 0
        for i in range(len(contours)):
            if cv.contourArea(contours[i]) > max_area:
                max_area = cv.contourArea(contours[i])
                max_index = i

        for eps in np.linspace(0.001, 0.05, 10):
            peri = cv.arcLength(contours[max_index], True)
            approx = cv.approxPolyDP(contours[max_index], eps * peri, True)
            if len(approx) == 4:
                chessboard_contour = approx
                break

        # for contour in contours:
        #     epsilon = 0.02 * cv.arcLength(contour, True)
        #     approx = cv.approxPolyDP(contour, epsilon, True)
        #     if len(approx) == 4:  # Quadrilateral
        #         area = cv.contourArea(contour)
        #         aspect_ratio = float(max(cv.boundingRect(contour)[2:])) / min(
        #             cv.boundingRect(contour)[2:]
        #         )
        #         if (
        #                 area > 5000 and 0.8 < aspect_ratio < 1.2
        #         ):  # Reasonable size and aspect ratio
        #             chessboard_contour = approx
        #             break

        #title = "detected"
        #if chessboard_contour is None:
        #    title = "not " + title


        if chessboard_contour is None:
            self.error = "Cannot find chessboard in the image. Make sure that it is completely visible."

            #cv.drawContours(img, contours, -1, (255, 0, 0), 2)
            #cv.imshow(title, img)

            #cv.waitKey(0)
            #cv.destroyAllWindows()
            return None, None

        # Apply perspective transformation
        pts_src = order_points(
            np.array([point[0] for point in chessboard_contour], dtype="float32")
        )
        pts_dst = np.array([[0, 0], [300, 0], [300, 300], [0, 300]], dtype="float32")
        matrix = cv.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv.warpPerspective(img, matrix, (300, 300))

        #cv.imshow("calibrate", warped)
        #cv.waitKey(0)
        #cv.destroyAllWindows()

        return matrix, warped

    def _process_image(self):
        img = self.image
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        hsv = cv.medianBlur(hsv, 5)


        # Create masks
        mask_blue = get_mask_blue(hsv)
        mask_red = get_mask_red(hsv)

        # Find contours
        contours_blue, _ = cv.findContours(
            mask_blue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        contours_red, _ = cv.findContours(
            mask_red, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        #cv.drawContours(img, contours_blue, -1, (255, 0, 0), 2)
        #cv.drawContours(img, contours_red, -1, (0, 0, 255), 2)
        #cv.imshow("Contours", img)
        #cv.waitKey(0)
        #cv.destroyAllWindows()

        # Filter contours
        contours_blue_filtered = filter_contours_by_size_and_shape(
            contours_blue#, min_size=300, max_size=2000
        )
        contours_red_filtered = filter_contours_by_size_and_shape(contours_red)

        #cv.drawContours(img, contours_blue_filtered, -1, (255, 0, 0), 2)
        #cv.drawContours(img, contours_red_filtered, -1, (0, 0, 255), 2)
        #cv.imshow("Filtered Contours", img)
        #cv.waitKey(0)
        #cv.destroyAllWindows()

        # Define grid size and calculate cell dimensions
        grid_size = 8
        cell_size_x = img.shape[1] / grid_size
        cell_size_y = img.shape[0] / grid_size

        # Process blue contours
        for contour in contours_blue_filtered:
            pos = get_position(contour, cell_size_x, cell_size_y)
            if pos:
                self.positions["blue"].append(pos)

        # Process red contours
        for contour in contours_red_filtered:
            pos = get_position(contour, cell_size_x, cell_size_y)
            if pos:
                self.positions["red"].append(pos)

        if self.positions["blue"]:
            for col, row, cX, cY in self.positions["blue"]:
                cell_x, cell_y = get_cell_based_position(
                    cX, cY, cell_size_x, cell_size_y
                )
                self.piece_positions[(col, row)] = (cX, cY, cell_x, cell_y)
                # image_center_x = cX#int((col + 0.5) * cell_size_x)
                # image_center_y = cY#int((row + 0.5) * cell_size_y)
                # cv.circle(img, (image_center_x, image_center_y), 15, (255, 0, 0), -1)
                # cv.putText(
                #     img,
                #     "Blue",
                #     (image_center_x - 20, image_center_y - 20),
                #     cv.FONT_HERSHEY_SIMPLEX,
                #     0.5,
                #     (255, 0, 0),
                #     1,
                # )

        if self.positions["red"]:
            for col, row, cX, cY in self.positions["red"]:
                cell_x, cell_y = get_cell_based_position(
                    cX, cY, cell_size_x, cell_size_y
                )
                self.piece_positions[(col, row)] = (cX, cY, cell_x, cell_y)
                # image_center_x = cX#int((col + 0.5) * cell_size_x)
                # image_center_y = cY#int((row + 0.5) * cell_size_y)
                # cv.circle(img, (image_center_x, image_center_y), 15, (0, 0, 255), -1)
                # cv.putText(
                #     img,
                #     "Red",
                #     (image_center_x - 20, image_center_y - 20),
                #     cv.FONT_HERSHEY_SIMPLEX,
                #     0.5,
                #     (0, 0, 255),
                #     1,
                # )

        # Update board matrix
        self._update_matrix(self.positions["blue"], 2)  # 2 for blue
        self._update_matrix(self.positions["red"], 1)  # 1 for red

        for key, value in self.piece_positions.items():
            x, y = key
            _, _, cell_x, cell_y = value
            #print(f"[{x}, {y}]: {cell_x, cell_y}")
            if (
                    cell_x < self.padding
                    or cell_x > 1 - self.padding
                    or cell_y < self.padding
                    or cell_y > 1 - self.padding
            ):
                self.error = f"Piece center at {x}, {y} is too close to the edge or the cell ({cell_x, cell_y})"
                break

    def _update_matrix(self, positions, piece_type):
        """
        Updates the board matrix based on detected positions.
        """
        for col, row, _, _ in positions:
            self.board_matrix[row][col] = piece_type

    def get_captured_box(self, image_path):
        # Load the image
        image = image_path
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Convert to RGB for correct display

        # Step 1: Convert the image to HSV to detect white color
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        # Define the range for white color in HSV
        lower_white = np.array([0, 0, 200], dtype=np.uint8)
        upper_white = np.array([180, 50, 255], dtype=np.uint8)

        # Create a binary mask for white color
        mask_white = cv.inRange(hsv_image, lower_white, upper_white)

        # Step 2: Find contours in the mask
        contours, _ = cv.findContours(mask_white, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # cv.drawContours(image_rgb, contours, -1, (0, 255, 0), 2)
        # cv.imshow("Contours", image_rgb)
        # cv.waitKey(0)

        # Step 3: Sort contours by area and get the second and third largest ones
        sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)[1:2]

        # Step 4: Calculate the centers of the selected contours
        boxes_centers = []
        for contour in sorted_contours:
            M = cv.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                boxes_centers.append((cx, cy))
        # cv.circle(image_rgb, boxes_centers[0], 5, (255, 0, 0), -1)
        # cv.circle(image_rgb, boxes_centers[1], 5, (255, 0, 0), -1)
        # cv.imshow("Detected boxes", image_rgb)
        # cv.waitKey(0)
        if len(boxes_centers) > 0:
            return boxes_centers[0]
        else:
            self.error = "Box for captured pieces not found in the image."
            return None

    def get_board_state(self):
        """
        Returns the current board state.
        """
        if not self.rotate:
            return self.board_matrix.tolist()
        return np.rot90(self.board_matrix, -1).tolist()

    def can_extract_board_state(self):
        """
        Checks if the board state can be extracted.
        """
        return self.error is None

    def get_temp_centers(self, show_image=False):
        """
        Calculates the centers of grid cells and optionally saves the image with centers marked.
        """
        image = self.image
        grid_size = 8
        cell_size_y = image.shape[1] / grid_size
        cell_size_x = image.shape[0] / grid_size
        centers = []

        for row in range(grid_size):
            row_centers = []
            for col in range(grid_size):
                cx = int((col + 0.5) * cell_size_x)
                cy = int((row + 0.5) * cell_size_y)
                if (col, row) in self.piece_positions.keys():
                    cx, cy, _, _ = self.piece_positions[(col, row)]
                row_centers.append((cx, cy))

                if show_image:
                    cv.circle(image, (cx, cy), 5, (0, 255, 0), -1)

            centers.append(row_centers)

        if show_image:
            cv.imshow("grid&pieces_center", image)
            cv.waitKey(0)
            cv.destroyAllWindows()
        return centers

    def get_centers(self):
        """
        Maps grid centers from the warped image back to the original image coordinates.
        """
        centers = self.get_temp_centers()
        centers_array = np.array(centers, dtype=np.float32).reshape(-1, 1, 2)
        transformation_matrix_inv = np.linalg.inv(self.transformation_matrix)
        original_coordinates = cv.perspectiveTransform(
            centers_array, transformation_matrix_inv
        )

        # Convert the coordinates to integers and return the result
        shaped = original_coordinates.reshape(8, 8, 2)
        arr_tuples = np.empty((8, 8), dtype=object)
        for i in range(8):
            for j in range(8):
                arr_tuples[i, j] = (round(shaped[i, j, 0]), round(shaped[i, j, 1]))
        if not self.rotate:
            return arr_tuples
        return np.rot90(arr_tuples, -1)

    def get_new_king_position(self):
        return self.new_king_position

    def calculate_new_king_position(self, red_king):
        # Load the image
        image = self.original_image

        # Convert to grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Apply CLAHE for better contrast
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Blur the image to reduce noise
        gray_blurred = cv.GaussianBlur(gray, (5, 5), 0)

        # Detect edges using Canny
        edges = cv.Canny(gray_blurred, 180, 220)

        # Clean up the edges using morphological operations
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        edges_cleaned = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv.findContours(edges_cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Sort contours by area and get the three largest ones
        sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)[:3]

        # Fill the largest contours with white in the original image
        for contour in sorted_contours:
            cv.drawContours(image, [contour], -1, (255, 255, 255), thickness=cv.FILLED)

        # Convert the image to HSV color space
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        mask_red = get_mask_red(hsv_image)
        mask_blue = get_mask_blue(hsv_image)

        # Function to find the centers of detected contours
        def find_centers(mask):
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            centers = []
            for contour in contours:
                if cv.contourArea(contour) > 100:  # Filter small contours
                    M = cv.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        centers.append((cx, cy))
            return centers

        # Get centers of red and blue pieces
        red_centers = find_centers(mask_red)
        blue_centers = find_centers(mask_blue)

        # Draw circles on the detected centers
        for center in red_centers:
            cv.circle(image, center, 10, (0, 0, 255), -1)  # Red circle

        for center in blue_centers:
            cv.circle(image, center, 10, (255, 0, 0), -1)  # Blue circle

        # cv.imshow("Processed Image", image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        # print("Red Centers:", red_centers)
        # print("Blue Centers:", blue_centers)

        if red_king:
            if len(red_centers) < 1:
                self.error = "Cannot find new king on the image. Make sure it is visible."
                return None
            return red_centers[0]
        else:
            if len(blue_centers) < 1:
                self.error = "Cannot find new king on the image. Make sure it is visible."
                return None
            return blue_centers[0]

    def get_captured_box_pos(self):
        """
        Get a position of the center of the red captured pieces box.
        :return: (box_x, box_y) image coordinates of the center of the red captured pieces box
        """
        return self.box

    def get_board_angle(self):
        centers = self.get_centers()
        top_left = centers[0][0]
        top_right = centers[0][7]
        angle = np.degrees(np.arctan2(top_right[1] - top_left[1], top_right[0] - top_left[0]))
        angle_max_90 = angle % 90
        other_direction = angle_max_90 - 90
        return angle_max_90 if angle_max_90 < abs(other_direction) else other_direction

    def _get_first_empty_index(self):
        empty_index = None
        for x in range(8):
            for y in range(8):
                if self.board_matrix[x][y] == 0:
                    empty_index = (x, y)
                    break
            if empty_index is not None:
                break
        return empty_index

    def _set_rotation(self):
        empty_index = self._get_first_empty_index()
        if empty_index is None:
            self.rotate = False
            pass
        row, col = empty_index
        gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)

        # Define grid size and calculate cell dimensions
        grid_size = 8
        cell_size_x = gray.shape[1] // grid_size
        cell_size_y = gray.shape[0] // grid_size

        gray_cropped = gray[row * cell_size_y : (row + 1) * cell_size_y, col * cell_size_x : (col + 1) * cell_size_x]
        roi_mean = gray_cropped.mean()
        self.rotate = (roi_mean > 180) != (row + col % 2 == 1)
        pass

# img_path = "img_9.jpg"
# image = cv.imread(img_path)
# if __name__ == "__main__":
#     detector = ImageDetector(image)
#     print("Board State:")
#     for row in detector.get_board_state():
#         print(row)
#     # print(detector.can_extract_board_state())
#     detector.get_temp_centers(show_image=False)
#     # for row in detector.get_centers():
#     #     print(row)
#     print(detector.get_red_captured_box_pos())
#     print(detector.get_blue_captured_box_pos())
#     detector.get_new_king_position(red_king=True)
