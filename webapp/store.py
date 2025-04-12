from collections import OrderedDict


class DrawingStore:
    def __init__(self, max_drawings: int = 10):
        self.drawings = OrderedDict()
        self.max_drawings = max_drawings

    def store(self, drawing_id: str, drawing: str):
        if len(self.drawings) >= self.max_drawings:
            self.drawings.popitem(last=False)

        self.drawings[drawing_id] = drawing

    def get(self, drawing_id: str):
        return self.drawings.get(drawing_id)

    def remove(self, drawing_id: str):
        self.drawings.pop(drawing_id, None)

    def contains(self, drawing_id: str):
        return drawing_id in self.drawings


drawing_store = DrawingStore()
