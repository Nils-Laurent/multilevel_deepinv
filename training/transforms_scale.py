from multilevel.info_transfer import DownsamplingTransfer, BlackmannHarris


class ToCoarserScale:
    def __init__(self, def_filter=BlackmannHarris()):
        self.def_filter = def_filter

    def __call__(self, pic):
        p2 = pic.unsqueeze(0)
        cit_op = DownsamplingTransfer(p2, self.def_filter)
        return cit_op.projection(p2)[0, ::]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
