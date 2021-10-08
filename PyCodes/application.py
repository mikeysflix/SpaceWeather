from subapp_cme_analysis import *
from subapp_flare_analysis import *
from subapp_cme_and_flare_analysis import *
from subapp_cme_and_flare_and_sunspot_analysis import *
from subapp_active_region_analysis import *
from example_viewer import *

## specify directory to save figures/results
savedir = '/Users/.../Desktop/SpaceWeather/Figures/' # None

## specify directory with .fits files
fitsdir = '/Users/.../Desktop/SpaceWeather/Data/'

## initialize cmes
cmes = CoronalMassEjections()
cmes.load_observed_data() # '/Users/.../Desktop/SpaceWeather/Data/univ_all.txt')

flares = SolarFlares()
flares.load_observed_data() # '/Users/.../Desktop/SpaceWeather/Data/hessi_flare_list.txt')

sunspots = Sunspots()
sunspots.load_observed_data() # '/Users/.../Desktop/SpaceWeather/Data/SN_d_tot_V2.0.txt')

run_cme_analysis = True # True
run_flare_analysis = False # True
run_cme_and_flare_analysis = False # True
run_cme_and_flare_and_sunspot_analysis = False # True
run_active_region_analysis = False # True
run_example_figures = False # True

if __name__ == '__main__':

    ## reproducible random-state
    np.random.seed(0)

    ## run cme analyses
    if run_cme_analysis:
        perform_cme_analysis(
            cmes=cmes,
            raw=False, # False,
            regular=True, # False
            savedir=savedir)

    ## run solar flare analyses
    if run_flare_analysis:
        perform_flare_analysis(
            flares=flares,
            # raw=True,
            regular=True,
            savedir=savedir)

    if run_cme_and_flare_analysis:
        perform_cme_and_flare_analysis(
            cmes=cmes,
            flares=flares,
            solar_cycles=([23, 24],),
            speed_type='second order initial speed',
            energy_type='high energy',
            savedir=savedir)

    if run_cme_and_flare_and_sunspot_analysis:
        perform_cme_and_flare_and_sunspot_analysis(
            cmes=cmes,
            flares=flares,
            sunspots=sunspots,
            solar_cycles=([23, 24],),
            speed_type='second order initial speed',
            energy_type='high energy',
            savedir=savedir)

    ## run AR analysis
    if run_active_region_analysis:
        perform_active_region_analysis(
            fitsdir=fitsdir,
            savedir=savedir,
            convert_to_gray_scale=True,
            convert_to_false_color=True,
            animate_gray_scale_timelapse=True,
            animate_false_color_timelapse=True,
            parameters=('pixel peak', 'pixel trough', 'pixel difference', 'pixel total', 'pixel mean', 'pixel median', 'pixel standard deviation'),
            save=True)

    ## run example figures
    if run_example_figures:
        visualizer = ExampleViewer(
            savedir=savedir)
        visualizer.view_frechet_distribution(
            sharex=True,
            sharey=True,
            collapse_x=True,
            collapse_y=True,
            figsize=(12, 7),
            layout=None,
            save=True)
        visualizer.view_example_clusters(
            figsize=(12, 7),
            save=True)






##
