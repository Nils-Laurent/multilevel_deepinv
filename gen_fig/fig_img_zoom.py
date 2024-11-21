import matplotlib.pyplot as plt
import matplotlib.patches as patches

def img_zoom_gen(file_name, ext, area):
    image = plt.imread(file_name + "." + ext)
    width = 1024
    height = 1024

    # Define the region to zoom in
    zoom_start = (area[0], area[1])  # Start of the zoom region (x, y)
    zoom_width = area[2]  # Width of the zoom region
    zoom_height = area[3]  # Height of the zoom region

    # Create the main figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')

    # Show the main image
    ax.imshow(image, cmap='gray')

    # Highlight the zoomed region with a rectangle
    rect = patches.Rectangle(
        zoom_start, zoom_width, zoom_height,
        linewidth=1, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)

    # Create an inset axes for the zoomed-in region
    # [x, y, width, height] in figure coordinates
    # x is left-right
    # y is bottom-up
    zx, zy, zw, zh = [0.5, 0.0, 0.5, 0.5]
    zoom_ax = fig.add_axes([zx, zy, zw, zh])

    zoom_ax.imshow(image[
                   zoom_start[1]:zoom_start[1] + zoom_height,
                   zoom_start[0]:zoom_start[0] + zoom_width
                   ], cmap='gray')
    #zoom_ax.set_title("Zoomed In")
    zoom_ax.axis('off')  # Turn off the axes for the zoomed image

    # Add a rectangle around the zoomed-in image in the inset
    rect2 = patches.Rectangle(
        (zx * width, (zy + zh) * height), zw * width, zh * height,
        linewidth=4, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect2)

    # Remove any extra white space around the figure
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    output_filename = "zoom_" + file_name + "." + ext

    # dpi = pixels / figure size in inches
    fig.savefig(output_filename, dpi=256/6, bbox_inches='tight', pad_inches=0)

    # Display the result
    #plt.show(block=False)

def zoomed_images(pb, k, init=True):
    zoom_area = {
        '0': [40, 780, 80, 80],
        '1': [50, 800, 80, 80],
        '2': [380, 190, 80, 80],
        '3': [720, 20, 80, 80],
    }

    win = 'sinc'
    level = '4'
    names = [
        'ct_cset_' + k + '_n0.1_' + pb,
    ]

    mlevel_suffix = level + "L" + win

    methods = [
        'x_truth',
        'y',
        'DPIR_Long',
        'FB_TV_ML' + '_' + mlevel_suffix,
        'PnP',
    ]

    if init == True:
        methods.append('PnP_ML_INIT' + '_' + mlevel_suffix)
    else:
        methods.append('PnP_ML' + '_' + mlevel_suffix)

    for n in names:
        for m in methods:
            file_n = n + '_' + m
            print(file_n)
            img_zoom_gen(file_n, ext="png", area=zoom_area[k])


if __name__ == '__main__':
    k = '1'
    zoomed_images('inpainting', k)
    zoomed_images('demosaicing', k)
    zoomed_images('blur', k)

