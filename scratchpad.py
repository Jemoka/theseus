from theseus.models import SUPPORTED_CHIPS
from theseus.training.utils import estimate_max_batch_size

h200 = SUPPORTED_CHIPS["h200"]
estimate_max_batch_size(1_900_000_000, 512, h200)
