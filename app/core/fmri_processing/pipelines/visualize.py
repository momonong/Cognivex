import nibabel as nib
from nilearn import plotting
import os


def visualize_activation_map(
    activation_path: str,
    output_path: str,
    threshold: float = 0.1,
    title: str = "Activation Map"
) -> None:
    """
    Visualize and save a statistical activation map using Nilearn.

    Args:
        activation_path (str): Path to the NIfTI activation map.
        output_path (str): Path to save the generated figure.
        threshold (float): Minimum activation threshold for visualization.
        title (str): Title shown in the figure.

    Returns:
        None
    """
    activation_img = nib.load(activation_path)

    # Plot using mosaic view
    display = plotting.plot_stat_map(
        activation_img,
        display_mode="mosaic",
        threshold=threshold,
        title=title
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    display.savefig(output_path)
    # print(f"Activation map saved as {output_path}")
    
    return output_path

if __name__ == "__main__":
    visualize_activation_map(
        activation_path="output/capsnet/resampled/module_test/module_test_resampled.nii.gz",
        output_path="figures/capsnet/module_test/activation_map_mosaic.png",
        threshold=0.1,
        title="Activation Map (sub-14 conv3)"
    )
