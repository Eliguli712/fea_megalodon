import drawsvg as draw

__all__ = ["TileDecoratorFish"]

class TileDecoratorFish:
    def __init__(self, p1=None, p2=None, q=None):
        self.p1 = p1
        self.p2 = p2
        self.q  = q

    @staticmethod
    def _xy(v):
        if hasattr(v, "x") and hasattr(v, "y"):
            return float(v.x), float(v.y)
        if hasattr(v, "z"):
            z = v.z
            return float(z.real), float(z.imag)
        if isinstance(v, complex):
            return float(v.real), float(v.imag)
        return float(v[0]), float(v[1])

    def to_drawables(self, tile=None, layer=0, **kwargs):
        poly = tile.to_polygon()
        verts = getattr(poly, "vertices", None) or getattr(poly, "pts", None)
        if verts is None:
            raise AttributeError("tile.to_polygon() has no vertices/pts")

        pts = [self._xy(v) for v in verts]

        if layer == 0:
            fill, stroke, sw = "rgba(20,20,30,0.08)", "rgba(0,0,0,0.25)", 0.006
        elif layer == 1:
            fill, stroke, sw = "rgba(60,80,160,0.10)", "rgba(0,0,0,0.28)", 0.006
        else:
            fill, stroke, sw = "none", "rgba(0,0,0,0.35)", 0.007

        path = draw.Path(fill=fill, stroke=stroke, stroke_width=sw)
        x0, y0 = pts[0]
        path.M(x0, y0)
        for x, y in pts[1:]:
            path.L(x, y)
        path.Z()
        return (path,)
