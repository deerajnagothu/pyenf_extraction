from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt

image = img_as_float(io.imread("../Recordings/nobody_original_frame.png"))
# loop over the number of segments
for numSegments in (100,500):
    # apply SLIC and extract (approximately) the supplied number
    # of segments
    segments = slic(image, n_segments=numSegments, sigma=5, start_label=1)
    # show the output of SLIC
    fig = plt.figure("Superpixels -- %d segments" % (numSegments))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segments, color=(1, 0, 0)))
    plt.axis("off")
# show the plots
plt.show()
