from theseus.models import SUPPORTED_HARDWARE
from theseus.training.utils import estimate_max_batch_size

h200 = SUPPORTED_HARDWARE["h200"]
estimate_max_batch_size(1_900_000_000, 512, h200)
