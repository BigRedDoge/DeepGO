

class FrameCrop:
    def __init__(self, frame, x, y, w, h):
        self.frame = frame
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def crop(self):
        return self.frame[self.y:self.y + self.h, self.x:self.x + self.w]
    