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

def zoomed_images(pb, gen_grid=False):
    w = 80
    h = 80
    zoom_area = {
        #     off_x, off_y, w, h
        '59': [70, 500, w, h],
        '66': [200, 775, w, h],
        '135': [150, 750, w, h],
    }
    img_id = zoom_area.keys()

    win = 'sinc'
    level = '4'

    mlevel_suffix = level + "L" + win

    methods = [
        'x_truth',
        'y',
        'DPIR_Long',
        'FB_TV_ML' + '_' + mlevel_suffix,
        'PnP',
        'PnP_ML_INIT' + '_' + mlevel_suffix,
        #'PnP_ML_DnCNN_init' + '_' + mlevel_suffix,
        'PnP_ML_SCUNet_init' + '_' + mlevel_suffix,
        'PnP_prox_ML_INIT' + '_' + mlevel_suffix,
        'PnP_prox',
    ]

    scale = 3
    fig, axarr = plt.subplots(nrows=1, ncols=len(img_id), figsize=(scale*len(img_id), scale*1))

    for m in methods:
        col_k = -1
        for k in img_id:
            col_k += 1
            prefix_pb = 'eq_ct_LIU4K-v2_' + k + '_n0.1_' + pb
            file_n = prefix_pb + '_' + m
            print(file_n)
            if gen_grid is True:
                img = plt.imread('zoom_' + file_n + ".png")
                axarr[col_k].imshow(img)
                axarr[col_k].axis('off')
            else:
                img_zoom_gen(file_n, ext="png", area=zoom_area[k])

        if gen_grid is True:
            plt.tight_layout()
            plt.subplots_adjust(wspace=0, hspace=0)

            png_name = 'zoom_mult_ct_cset_n0.1_' + pb + '_' + m
            fig.savefig(png_name+".png", dpi=256 / 6, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    #zoomed_images('inpainting0.8')
    #zoomed_images('inpainting0.9')
    #zoomed_images('blur')
    zoomed_images('inpainting0.5', gen_grid=False)
    zoomed_images('demosaicing', gen_grid=False)
    #zoomed_images('inpainting0.5', gen_grid=True)
    #zoomed_images('demosaicing', gen_grid=True)
