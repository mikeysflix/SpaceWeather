from histogram_configuration import *

def example_histogram(data_id):
    """
    data_id:
        'normal' or 'datetime'
    """

    if data_id == 'normal':
        x = np.random.normal(50, 10, 5000)
        histogram = Histogram(x)
        histogram.update_edges(
            lbin=0,
            rbin=100,
            # nbins=10,
            wbin=10)
        histogram.update_counts()

    elif data_id == 'datetime':

        energy = np.arange(0, 201, 2).astype(int)
        elapsed = np.arange(energy.size).astype(int)
        is_event = np.random.choice(
            np.array([True, False]),
            size=elapsed.size,
            replace=True,
            p=(0.7, 0.3))
        energy[np.invert(is_event)] = 0
        first_dt = datetime.datetime(1999, 1, 1, 0, 0, 0)
        dts = np.array([first_dt + relativedelta(days=i)
            for i in range(elapsed.size)])
        data = {
            'energy' : energy,
            'elapsed' : elapsed,
            'datetime' : dts}
        histogram = TemporalHistogram(data)
        histogram.update_edges(
            lbin=dts[0],
            rbin=dts[-1],
            wbin=10,
            time_step='day')
        histogram.update_counts()

    else:
        raise ValueError("invalid data_id: {}".format(data_id))

    print(histogram)

# example_histogram('normal')
# example_histogram('datetime')





##
