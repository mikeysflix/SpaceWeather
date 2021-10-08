from visual_configuration import *

class ExampleViewer(VisualConfiguration):

    def __init__(self, savedir=None):
        super().__init__(savedir=savedir)

    def view_frechet_distribution(self, sharex=False, sharey=False, collapse_x=False, collapse_y=False, figsize=None, save=False, layout='single'):
        permutable_layouts = ['horizontal', 'vertical']
        if layout is None:
            self.view_layout_permutations(
                f=self.view_frechet_distribution,
                layouts=permutable_layouts,
                sharex=sharex,
                sharey=sharey,
                collapse_x=collapse_x,
                collapse_y=collapse_y,
                figsize=figsize,
                save=save)
        elif layout == 'single':
            visualizer = ExampleViewer(
                savedir=self.savedir)
            for layout in permutable_layouts:
                visualizer.view_frechet_distribution(
                    sharex=sharex,
                    sharey=sharey,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y,
                    figsize=figsize,
                    save=save,
                    layout=layout)
        else:
            ## verify user input
            if layout not in ('horizontal', 'vertical'):
                raise ValueError("invalid layout for this method: {}".format(layout))
            ## initialize distribution parameters
            xs = np.arange(0, 5.001, .005)
            x = np.delete(xs, 0)
            ys = [0]
            alpha = np.array([1, 1, 2, 2, 3, 3])
            mu = np.zeros(alpha.size, dtype=int)
            sigma = np.array([1, 2, 1, 2, 1, 2])
            facecolors = ('red', 'green', 'blue', 'orange', 'purple', 'black')
            ## initialize plot
            shared_xlabel = 'x'
            kws = self.get_number_of_figure_rows_and_columns(nseries=2, layout=layout)
            fig, axes = plt.subplots(figsize=figsize, **kws)
            for a, m, s, fc in zip(alpha, mu, sigma, facecolors):
                tmp = (x - m)/s
                for i, ax in enumerate(axes.ravel()):
                    if i == 0:
                        y = (a/s) * tmp**(-1 - a) * np.exp(-1 * (tmp ** -a))
                        label = r'$\alpha = {}, \mu = {}, \sigma = {}$'.format(a, m, s)
                        formula = r'$PDF(x | \alpha, \mu, \sigma) = \frac{\alpha}{\sigma} (\frac{x-\mu}{\sigma})^{-1-\alpha} e^{- (\frac{x-\mu}{\sigma})^{-\alpha}}$'
                        title = 'Probability Density'
                    else:
                        y = np.exp(-1 * (tmp ** -a))
                        label = None
                        formula = r'$CDF(x | \alpha, \mu, \sigma) = e^{- (\frac{x-\mu}{\sigma})^{-\alpha}}$'
                        title = 'Cumulative Density'
                    ax.plot(
                        x,
                        y,
                        color=fc,
                        alpha=0.5,
                        label=label)
                    ax.set_xlabel(shared_xlabel, fontsize=self.labelsize)
                    ax.set_ylabel(formula, fontsize=self.labelsize)
                    ax.set_title(title, fontsize=self.titlesize)
                    ys.append(np.max(y))
            ## update axes
            self.share_axes(
                axes=axes,
                layout=layout,
                xs=xs,
                ys=ys,
                sharex=sharex,
                sharey=sharey,
                xticks=True,
                yticks=True,
                xlim=True,
                ylim=True,
                xlabel=shared_xlabel,
                collapse_x=collapse_x,
                collapse_y=collapse_y)
            ## show legend
            handles, labels = [], []
            for ax in axes.ravel():
                self.apply_grid(ax)
                _handles, _labels = ax.get_legend_handles_labels()
                handles.extend(_handles)
                labels.extend(_labels)
            self.subview_legend(
                fig=fig,
                ax=axes.ravel()[0],
                handles=handles,
                labels=labels,
                textcolor=True,
                bottom=0.2,
                ncol=len(handles)//2)
            ## update title
            fig.suptitle(r'Fr$\acute{e}$chet Distribution', fontsize=self.titlesize)
            ## show / save
            if save:
                savename = 'example_frechet_distribution_{}'.format(layout)
            else:
                savename = None
            self.display_image(fig, savename=savename)

    def view_example_clusters(self, figsize=None, save=False):
        ## initialize cluster configuration parameters
        time_threshold = 5
        clusters = np.array([np.array([2, 3, 5]), np.array([11, 13, 17]), np.array([22, 26])])
        events = np.concatenate(clusters, axis=0)
        facecolors = ('darkorange', 'steelblue', 'purple')
        markers = ('o', '*', '^')
        xticks = np.arange(np.max(events) + 2)
        arrowprops = dict(facecolor='gray', arrowstyle='<->')
        ## initialize plot
        fig, ax = plt.subplots(figsize=figsize)
        for i, (cluster, facecolor, marker) in enumerate(zip(clusters, facecolors, markers)):
            ## label intra-times and intra-durations
            intra_times = np.diff(cluster)
            intra_duration = cluster[-1] - cluster[0]
            intra_times_label = "Intra-Times = {$" + ",".join(str(int(intra_time)) for intra_time in intra_times) + "$} hours"
            intra_duration_label = "Intra-Duration = ${}$ hours".format(intra_duration)
            cluster_id_label = "Cluster #${}$".format(i+1)
            label = '{}\n{}\n{}'.format(cluster_id_label, intra_duration_label, intra_times_label)
            ax.scatter(
                cluster,
                np.ones(cluster.size),
                color=facecolor,
                marker=marker,
                s=5,
                label=label)
        ## label inter-durations
        for prev_cluster, next_cluster in zip(clusters[:-1], clusters[1:]):
            xi, xf = prev_cluster[-1], next_cluster[0]
            inter_duration = xf - xi
            ax.annotate(
                '',
                xy=(xi, 0.95),
                xytext=(xf, 0.95),
                fontsize=self.textsize,
                horizontalalignment='center',
                verticalalignment='center',
                arrowprops=arrowprops)
            ax.text(
                np.mean([xi, xf]),
                0.975,
                'Inter-Duration = ${}$ hours'.format(inter_duration),
                fontsize=self.textsize,
                horizontalalignment='center',
                verticalalignment='center')
        ## update axes
        ax.set_xlabel('Elapsed Time (hours)', fontsize=self.labelsize)
        ax.set_xticks(xticks, minor=True)
        ax.set_xticks(xticks[::5])
        ax.set_xlim([0, xticks[-1]])
        ax.set_ylim([0.85, 1.05])
        ax.set_yticks([])
        self.apply_grid(ax,
            which='major',
            axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ## show legend
        handles, labels = ax.get_legend_handles_labels()
        self.subview_legend(
            fig=fig,
            ax=ax,
            handles=handles,
            labels=labels,
            textcolor=True,
            facecolor='lightgray',
            bottom=0.2,
            ncol=len(handles),
            title='Time Threshold $T_C$ = ${}$ hours'.format(time_threshold))
        ## update title
        fig.suptitle('Example of Cluster Intra-Times, Intra-Durations, and Inter-Durations', fontsize=self.titlesize)
        ## show / save
        if save:
            savename = 'example_clusters'
        else:
            savename = None
        self.display_image(fig, savename=savename)





##
