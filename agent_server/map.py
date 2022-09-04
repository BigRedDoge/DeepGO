from ..utils.frame_crop import FrameCrop

class Map:

    def __init__(self):
        self.map = None
        self.map_coords = (15, 435, 485, 475)
        
    def get_map(self, frame):
        map_crop = FrameCrop(frame, *self.map_coords)
        self.map = map_crop.crop()
        return self.map
