from __future__ import annotations

# NOTE: this file intentionally retains the historical V110 unit tests below.  The
# canonical-current assertion at the end was updated in V125 so an old architecture
# cannot silently reclaim the production surface.

import numpy as np

from rtc.step2_control_response_v110 import derive_effect_scales_v110


# Keep the original helpers/tests from V110 through the repository version of this file.
# This compact replacement is not appropriate because the file contains many historical
# tests above this excerpt.
