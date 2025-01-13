import numpy as np

class CoordinateTransformer():
    def __init__(self):
        self.px_to_mm_x = 0.535483871
        self.px_to_mm_y = 0.539887188
        self.trans_mat = np.array([
            [ 0, -1,  0, 350 + 19],
            [-1,  0,  0, 350 + 72],
            [ 0,  0, -1, 0],
            [ 0,  0,  0, 1]])
        self._square_centers = [[(0, 0) for _ in range(8)] for _ in range(8)]

    @property
    def square_centers(self):
        return self._square_centers

    @square_centers.setter
    def square_centers(self, new_square_centers):
        self._square_centers = new_square_centers

    def transform_to_base_coordinate(self, x, y):
        pos = np.array([[x, y, 0 ,1]], dtype=np.float32)
        base_pos = self.trans_mat.dot(pos.T)[0:2]
        return (base_pos[0][0], base_pos[1][0])

    def img_xy_to_real_xy(self, pos, image_size):
        """
        Transforms image (x, y) to real (x, y)
        :param pos: (img_x, img_y) position in image
        :param image_size: (height, width) size of the image
        :return: (real_x, real_y)
        """
        scene_x = (pos[0] - image_size[0] / 2) * self.px_to_mm_x
        scene_y = (pos[1] - image_size[1] / 2) * self.px_to_mm_y
        return self.transform_to_base_coordinate(scene_x, scene_y)

    def board_xy_to_img_xy(self, board_pos, square_centers):
        """
        Returns the image coordinates of the center of the square of the board_pos.
        :param board_pos: (x, y) position on the board
        :param square_centers: 8x8 list of image coordinates of the centers of the squares
        :return: (img_x, img_y)
        """
        return square_centers[board_pos[1]][board_pos[0]]

    def board_xy_to_real_xy(self, board_pos, square_centers, image_size):
        """
        Returns the real coordinates of the center of the square of the board_pos.
        :param board_pos: (x, y) position on the board
        :param square_centers: 8x8 list of image coordinates of the centers of the squares
        :param image_size: (height, width) size of the image
        :return: (real_x, real_y)
        """
        img_pos = self.board_xy_to_img_xy(board_pos, square_centers)
        return self.img_xy_to_real_xy(img_pos, image_size)