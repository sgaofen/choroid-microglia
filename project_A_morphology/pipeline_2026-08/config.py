"""Dataset-specific constants shared by every script in this bundle.

**Before running on a new image set, check these two values.** They default to the
2026-08 10x set (0.164827 um/px, 14661 x 14661 px). Every micrometre, mm^2 and
per-mm^2 number in the pipeline scales off PIX_UM; IMG_PX is used to clip ROI
rectangles to the image. A wrong value produces wrong numbers with NO error and
NO crash - nothing downstream validates them.

PIX_UM  physical size of one pixel, micrometres. Read it from the acquisition
        metadata (tif_max/*.meta.json) of the new set, not from here.
IMG_PX  side length of the square scan in pixels. If your new scans are not
        square, set IMG_PX to the smaller side, or better, read the real shape
        from the TIF (`tifffile.memmap(path).shape`) in the script you are running.

Values are frozen for the 2026-08 delivery: every published number assumes them.
"""

PIX_UM = 0.164827
IMG_PX = 14661


# --- where the images live -------------------------------------------------
# Edit these two, or set the CHP_SRC / CHP_FJFF environment variables. Nothing
# else in the bundle hard-codes a data path.
#
#   SRC_TIF_MAX      max-projected TIFs. Read by server.py (to enumerate samples
#                    and as the fallback image source) and by stats_corrected.py
#                    (to build the tissue mask). Required.
#   SRC_FLATFIELDED  Fiji pseudo-flat-fielded TIFs. Only preprocess_clean_images.py
#                    reads these, and only when rebuilding clean/ from scratch.
import os

SRC_TIF_MAX = os.environ.get("CHP_SRC", "/path/to/source_raw/_work/tif_max")
SRC_FLATFIELDED = os.environ.get("CHP_FJFF", "/path/to/source_raw/_work/tif_fiji_flatfielded")
