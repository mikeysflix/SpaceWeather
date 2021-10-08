from data_processing import *

def perform_active_region_analysis(fitsdir, savedir=None, convert_to_gray_scale=False, convert_to_false_color=False, animate_gray_scale_timelapse=False, animate_false_color_timelapse=False, parameters=None, save=False):

    ars = ActiveRegions()
    ars.load_observed_data(
        fitsdir=fitsdir)
    ars.load_raw_series()
    if parameters is not None:
        if isinstance(parameters, str):
            parameters = [parameters]
        elif not isinstance(parameters, (tuple, list, np.ndarray)):
            raise ValueError("invalid type(parameters): {}".format(type(parameters)))
        for parameter in parameters:
            ars.view_pixel_data(
                parameters=parameter,
                savedir=savedir,
                figsize=(12, 7))
        if len(parameters) > 1:
            ars.view_pixel_data(
                parameters=parameters,
                savedir=savedir,
                figsize=(12, 7))
    if (convert_to_gray_scale or convert_to_false_color):
        ars.convert_fits_to_images(
            fitsdir=fitsdir,
            savedir=savedir,
            convert_to_gray_scale=convert_to_gray_scale,
            convert_to_false_color=convert_to_false_color,
            vmin=-60,
            vmax=60,
            img_extension='.png',
            save=True,
            figsize=(12, 7))
    if (animate_gray_scale_timelapse or animate_false_color_timelapse):
        if animate_gray_scale_timelapse:
            ars.animate_timelapse(
                img_dir=savedir,
                fps=5,
                show_gray_scale=True,
                show_false_color=False,
                img_extension='.png',
                mov_extension='.mp4', # '.mkv',
                codec='mpeg4',
                save=True)
            ars.animate_timelapse(
                img_dir=savedir,
                fps=5,
                show_gray_scale=True,
                show_false_color=False,
                img_extension='.png',
                mov_extension='.gif',
                codec='mpeg4',
                save=True)
        if animate_false_color_timelapse:
            ars.animate_timelapse(
                img_dir=savedir,
                fps=5,
                show_gray_scale=False,
                show_false_color=True,
                img_extension='.png',
                mov_extension='.mp4', # '.mkv',
                codec='mpeg4',
                save=True)
            ars.animate_timelapse(
                img_dir=savedir,
                fps=5,
                show_gray_scale=False,
                show_false_color=True,
                img_extension='.png',
                mov_extension='.gif',
                codec='mpeg4',
                save=True)







##
