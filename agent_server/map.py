from mmap import MADV_AUTOSYNC
from frame_crop import FrameCrop

class Map:

    def __init__(self):
        self.map = None
        self.map_coords = (0, 0, 0, 0)
        self.set_map()
        
    def get_map(self, frame):
        map_crop = FrameCrop(frame, *self.map_coords)
        self.map = map_crop.crop()
        return self.map
