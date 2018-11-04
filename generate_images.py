import argparse
import cv2
import matplotlib.pyplot as plt
import os


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def update_plots():
    global src_rect, crop
    src_rect = cv2.rectangle(src.copy(), (x, y), (x+w, y+h), (0, 255, 0), 2)
    crop = src[y:y+h, x:x+w]
    ax_list[0].imshow(src_rect[..., ::-1])
    ax_list[1].imshow(crop[..., ::-1])
    plt.draw()
    print "dimensions: %0.2f x %0.2f   ratio: h=w*%0.2f" % (w, h, ratio)


def onclick(event):
    global x, y
    x = int(event.xdata)
    y = int(event.ydata)
    update_plots()


def press(event):
    global x, y, w, h, ratio, ratio_locked, images, crop
    if event.key == 'right':
        x += translate
    elif event.key == 'left':
        x -= translate
    elif event.key == 'up':
        y -= translate
    elif event.key == 'down':
        y += translate

    if event.key == 'd':
        if ratio_locked:
            w += scale
            h = int(w * ratio)
        else:
            w += scale
    elif event.key == 'a':
        if ratio_locked:
            w -= scale
            h = int(w * ratio)
        else:
            w -= scale
    elif event.key == 'w':
        if ratio_locked:
            h += scale
            w = int(h / ratio)
        else:
            h += scale
    elif event.key == 's':
        if ratio_locked:
            h -= scale
            w = int(h / ratio)
        else:
            h -= scale
    elif event.key == 'r':
        if ratio_locked:
            ratio_locked = False
            print "free ratio"
        else:
            ratio_locked = True
            print "ratio locked: %0.2f" % ratio
    elif event.key == ' ':
        images += 1
        cv2.imwrite("%simage_%d.jpg" % (target_dir, images), crop)
        print "image saved!"
    ratio = h*1.0 / w
    update_plots()


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())

src = cv2.imread(args['image'], cv2.IMREAD_UNCHANGED)
x = src.shape[1] / 2
y = src.shape[0] / 2
w = 50
h = 50
translate = 2
scale = 2
ratio = 1.0
ratio_locked = False
images = 0
target_dir = 'cropped_images/'
ensure_dir(target_dir)

fig, _ax_list = plt.subplots(1, 2)
ax_list = _ax_list.ravel()
ax_list[0].set_title('ORIGINAL \n-arrow keys or click to navigate \n-WASD change dimensions \n-R to lock/unlock ratio \n-SPACE to save cropped image')
ax_list[1].set_title('CROPPED')

src_rect = cv2.rectangle(src.copy(), (x, y), (x+w, y+h), (0, 255, 0), 2)
crop = src[y:y+h, x:x+w]

fig.canvas.mpl_connect('key_release_event', press)
fig.canvas.mpl_connect('button_press_event', onclick)
plt.rcParams['keymap.save'] = ''
update_plots()
plt.show()
