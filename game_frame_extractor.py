from screen_crop import ScreenCrop


class GameFrameExtractor():
    # map_coords is a tuple of coordinates for the map
    # map_coords = (x, y, h, w)
    def __init__(self, map_coords):
        self.frame = None
        self.map_coords = map_coords

    def set_frame(self, frame):
        self.frame = frame

    def get_map(self, frame):
        self.set_frame(frame)
        if self.frame is None:
            return None
        return ScreenCrop(self.frame).crop(*self.map_coords)

    