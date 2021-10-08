from search_methods import *

def example_search(data_id, i):
    """
    data_id:
        'event' or 'cluster'

    i:
        i=0 ==> search data for one numerical parameter condition

        i=1 ==> search data for one string parameter condition and
                one numerical parameter condition, use any-criteria

        i=2 ==> search data for one string parameter condition and
                one numerical parameter condition, use all-criteria

        i=3 ==> search data for three numerical parameter conditions
                and one datetime parameter condition, use all-criteria
    """

    ## check inputs
    if data_id not in ('event', 'cluster'):
        raise ValueError("invalid data_id: {}".format(data_id))
    if i not in list(range(4)):
        raise ValueError("invalid i: {}".format(i))

    ## initialize sample data
    x = np.arange(10, dtype=int)
    y = x - np.median(x)
    z = np.square(y)
    s = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    dt = datetime.datetime(1990, 10, 28, 0, 0, 0)
    t = np.array([dt + datetime.timedelta(days=i) for i in range(x.size)])

    ## select search criteria
    if i == 0: ## search data for one numerical parameter condition
        search_kwargs = {
            'parameters' : 'x',
            'conditions' : 'greater than or equal',
            'values' : 5}
    elif i == 1: ## search data for one string parameter condition and one numerical parameter condition, use any-criteria
        search_kwargs = {
            'parameters' : ('s', 'x'),
            'conditions' : ('not equal', 'greater than'),
            'values' : ('h', 5),
            'apply_to' : 'any'}
    elif i == 2: ## search data for one string parameter condition and one numerical parameter condition, use all-criteria
        search_kwargs = {
            'parameters' : ('s', 'x'),
            'conditions' : ('not equal', 'greater than'),
            'values' : ('h', 5),
            'apply_to' : 'all'}
    elif i == 3: ## search data for three numerical parameter conditions and one datetime parameter condition, use all-criteria
        search_kwargs = {
            'parameters' : ('x', 'y', 'z', 't'),
            'conditions' : ('greater than or equal', 'greater than', 'less than', 'greater than'),
            'values' : (2, 2, 5, datetime.datetime(1990, 10, 31, 0, 0, 0)),
            'apply_to' : 'all'}
    else:
        raise ValueError("invalid i: {}".format(i))

    if data_id == 'event':

        ## consolidate data
        data = {
            'x' : x,
            'y' : y,
            'z' : z,
            't' : t,
            's' : s}

        ## get searched data and corresponding indices
        searcher = EventSearcher(data)
        events, indices = searcher.search_events(**search_kwargs)
        search_label = searcher.get_search_label(**search_kwargs)

        ## print-check results
        print("\n * ORIGINAL DATA")
        for k, v in data.items():
            print("\n .. {}:\n{}\n".format(k, v))
        print("\n * SEARCH LABEL")
        print(search_label)
        print("\n * INDICES:\n{}\n".format(indices))
        print("\n * EVENTS")
        for k, v in events.items():
            print("\n .. {}:\n{}\n".format(k, v))

    else:

        ## consolidate data
        split_loc = np.array([2, 3, 6, 8])
        data = {
            'x' : np.array(
                np.split(x, split_loc)),
            'y' : np.array(
                np.split(y, split_loc)),
            'z' : np.array(
                np.split(z, split_loc)),
            't' : np.array(
                np.split(t, split_loc)),
            's' : np.array(
                np.split(s, split_loc))}

        ## get searched data and corresponding indices
        searcher = ClusterSearcher(data)
        clusters, indices = searcher.search_events(**search_kwargs)
        search_label = searcher.get_search_label(**search_kwargs)

        ## print-check results
        print("\n * ORIGINAL DATA")
        for k, v in data.items():
            print("\n .. {}:\n{}\n".format(k, v))
        print("\n * SEARCH LABEL")
        print(search_label)
        print("\n * INDICES:\n{}\n".format(indices))
        print("\n * CLUSTERS")
        for k, v in clusters.items():
            print("\n .. {}:\n{}\n".format(k, v))

# example_search('event', 0)
# example_search('event', 1)
# example_search('event', 2)
# example_search('event', 3)
#
# example_search('cluster', 0)
# example_search('cluster', 1)
# example_search('cluster', 2)
# example_search('cluster', 3)








##
