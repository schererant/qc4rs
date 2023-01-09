import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import spectral
from plotly import tools as tls
import seaborn as sns


# TODO: rewrite
def build_dataset(img, gt, ignored_labels=None):
    """Build a dataset from an image and a ground truth.
    Args:
        img: 3D image
        gt: 2D ground truth
        ignored_labels (optional): list of labels to ignore

    Returns:
        dataset: 2D array of shape (n_samples, n_features)
        labels: 1D array of shape (n_samples)
    """
    samples = []
    labels = []
    indices_list = []
    # Check that image and ground truth have the same 2D dimensions
    assert img.shape[:2] == gt.shape[:2]
    
    for label in np.unique(gt):
        if label in ignored_labels:
            continue
        indices = np.nonzero(gt == label)
        indices_list.append(indices)
        samples += list(img[indices])
        labels += len(indices[0]) * [label]

    return np.asarray(samples), np.asarray(labels), indices_list


def pred_to_gt(indices_list, predictions, gt_shape):
    """Convert a prediction array to a ground truth array.
    Args:
        indices_list: list of indices of the samples
        predictions: 1D array of shape (n_samples)
        gt_shape: shape of the ground truth
    Returns:
        gt: 2D array of shape (n_rows, n_cols)
    """
    gt_pred = np.zeros(gt_shape)
    rolling = 0
    for indices in indices_list:
        for x,y in zip(indices[0], indices[1]):
            gt_pred[x,y] = predictions[rolling]
            rolling += 1
    return gt_pred


def send_to_visdom(mpl_fig, vis, fix_aspect=True):
    """Send a matplotlib figure to visdom.
    Args:
        mpl_fig: matplotlib figure
        vis: Visdom display
        fix_aspect: boolean
    """
    plotly_fig = tls.mpl_to_plotly(mpl_fig, resize=True)
    if fix_aspect:
        plotly_fig.layout.yaxis.scaleanchor='x'
        plotly_fig.layout.yaxis.scaleratio=1

    vis.plotlyplot(plotly_fig)
    # json.dump(plotly)
    # vis._send({'data':plotly_fig.data , 'layout':plotly_fig.layout})


#TODO: Alternative to matplotlib?
def show_img_gt(img, gt, label_values, rgb, vis):
    """Show an image and its ground truth.
    Args:
        img: 3D image
        gt: 2D ground truth
        label_values: list of class names
        rgb: boolean
        vis: Visdom display
    """
    # create handles for legend
    cmap = matplotlib.cm.get_cmap("tab20")
    handles = [matplotlib.patches.Patch(color=cmap(i), label=label_values[i]) for i in range(len(label_values))]

    # adjusts colormap according to classes present in label map
    def adjust_cmap(data, cmap_name):
        cmap = matplotlib.cm.get_cmap(cmap_name)
        colors = cmap(np.linspace(0, 1, cmap.N))
        adj_colors = colors[: np.max(data) + 1]
        return matplotlib.colors.ListedColormap(adj_colors)

    hypercube_rgb = spectral.get_rgb(img, bands=rgb)


    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    axes[0].set_title(
        "Hyperspectral Image ({}x{}x{})".format(*img.shape), fontsize=15, pad=20
    )
    axes[0].imshow(hypercube_rgb)
    axes[1].set_title("Ground Truth ({}x{})".format(*gt.shape), fontsize=15, pad=20)
    axes[1].imshow(gt, cmap=adjust_cmap(gt, "tab20"), interpolation="none")
    axes[1].legend(handles=handles, loc=(1.04, 0))

    vis.matplot(fig)

    # sendToVisdom(fig, vis)


#Taken from HyperspectalRepo
def explore_spectra(img, complete_gt, class_names, vis,
                      ignored_labels=None):
    """Plot sampled spectrums with mean + std for each class.

    Args:
        img: 3D hyperspectral image
        complete_gt: 2D array of labels
        class_names: list of class names
        ignored_labels (optional): list of labels to ignore
        vis : Visdom display
    Returns:
        mean_spectrums: dict of mean spectrum by class

    """
    mean_spectrums = {}
    for c in np.unique(complete_gt):
        # if c in ignored_labels:
        #     continue
        mask = complete_gt == c
        class_spectrums = img[mask].reshape(-1, img.shape[-1])
        step = max(1, class_spectrums.shape[0] // 100)
        fig = plt.figure()
        plt.title(class_names[c])
        # Sample and plot spectrums from the selected class
        for spectrum in class_spectrums[::step, :]:
            plt.plot(spectrum, alpha=0.25)
        mean_spectrum = np.mean(class_spectrums, axis=0)
        std_spectrum = np.std(class_spectrums, axis=0)
        lower_spectrum = np.maximum(0, mean_spectrum - std_spectrum)
        higher_spectrum = mean_spectrum + std_spectrum

        # Plot the mean spectrum with thickness based on std
        plt.fill_between(range(len(mean_spectrum)), lower_spectrum,
                         higher_spectrum, color="#3F5D7D")
        plt.plot(mean_spectrum, alpha=1, color="#FFFFFF", lw=2)

        if vis is not None:
            vis.matplot(plt)
        else:
            plt.show()
        mean_spectrums[class_names[c]] = mean_spectrum
    return mean_spectrums


def display_train_test_split(gt, train_gt, test_gt, ax0_name, ax1_name, label_values, vis):
    """
    Display the train and test split in a grid.
    Args:
        gt: 2D ground truth
        train_gt: 2D ground truth of the train split
        test_gt: 2D ground truth of the test split
        label_values: list of class names
        vis: Visdom display
    """

    # create handles for legend
    cmap = matplotlib.cm.get_cmap("tab20")
    handles = [matplotlib.patches.Patch(color=cmap(i), label=label_values[i]) for i in range(len(label_values))]

    # adjusts colormap according to classes present in label map
    def adjust_cmap(data, cmap_name):
        cmap = matplotlib.cm.get_cmap(cmap_name)
        colors = cmap(np.linspace(0, 1, cmap.N))
        adj_colors = colors[: np.max(data) + 1]
        return matplotlib.colors.ListedColormap(adj_colors)


    n_train = np.count_nonzero(train_gt)
    n_test = np.count_nonzero(test_gt)
    n_total = np.count_nonzero(gt)

    print("Train:  {} samples selected (over {})".format(n_train, n_total))
    print("Test:   {} samples selected (over {})".format(n_test, n_total))

    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    axes[0].set_title("{}".format(ax0_name), fontsize=15, pad=20)
    axes[0].imshow(train_gt, cmap=adjust_cmap(gt, "tab20"), interpolation="none")
    axes[1].set_title("{}".format(ax1_name), fontsize=15, pad=20)
    axes[1].imshow(test_gt, cmap=adjust_cmap(gt, "tab20"), interpolation="none")
    axes[1].legend(handles=handles, loc=(1.04, 0))

    if vis is not None:
        vis.matplot(fig)
    else:
        plt.show()
    # sendToVisdom(fig, vis)
     

def pretty_print_gram(gram_matrix, vis):
    """Pretty print a gram matrix.
    Args:
        gram_matrix: 2D array of shape (n_features, n_features)
    """
    ax = sns.heatmap(
    gram_matrix, 
    vmin=0, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    if vis is not None:
        vis.matplot(plt)
    else:
        plt.show()


def pretty_print_confusion_matrix(confusion_matrix, class_names, vis):

    plt.subplots(1, 1, figsize=(11, 7))
    sns.heatmap(
        confusion_matrix,
        fmt="g",
        xticklabels=class_names[1:],
        yticklabels=class_names[1:],
        annot=True,
    )
    plt.xlabel("True Label", fontsize=15)
    plt.ylabel("Predicted Label", fontsize=15)

    if vis is not None:
        vis.matplot(plt)
    else:
        plt.show()


def input_summary(arr):
    """Print summary of an input array.
    Args:
        arr: input array
    """
    print("----------------------------------------------------")
    print("Input Summary")
    print("----------------------------------------------------")
    print("Input array shape: {}".format(arr.shape))
    print("Shape: {}".format(arr.shape))
    print("Min: {}".format(np.min(arr)))
    print("Max: {}".format(np.max(arr)))
    print("Mean: {}".format(np.mean(arr)))
    print("Std: {}".format(np.std(arr)))
    print("----------------------------------------------------")


def visdom_text(vis, text):
    """Send text to Visdom.
    Args:
        vis: Visdom display
        text: text to send
    """
    vis.text(text)


#TODO: Make results a class
def print_results_terminal(results):
    """Print results in terminal.
    Args:
        results: dictionary of results
    """
    print('')
    print('-------------------------------------------------------------')
    print('(Training Data) Classification report')
    print('-------------------------------------------------------------')
    print(results['classification_report'])
