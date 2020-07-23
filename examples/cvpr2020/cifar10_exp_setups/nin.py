from .vggs import vgg_all
from lc.models.torch.nincif import nincif_bn

__all__ = ["nin_all"]

class nin_all(vgg_all):
  def __init__(self):
    super(nin_all, self).__init__("nin_all", nincif_bn(), 'nincif_bn')