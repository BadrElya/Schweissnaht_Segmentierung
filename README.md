# Instance Segmentation

This repository contains various scripts and tools for training and evaluating semantic and instance segmentation models. Below is an overview of the repository structure and its contents.

## Development Environment

For this project, the following versions were used in the Anaconda environment:

- conda version: 23.5.2
- conda-build version: 3.25.0
- python version: 3.11.3.final.0

Additionally, the following virtual packages were used:

- __archspec=1=x86_64
- __cuda=12.2=0
- __win=0=0

## Repository Structure

- **augmentedDataScripts/**: Contains Python scripts for augmenting training data.
- **customDataScripts/**: Contains scripts for processing training data as desired.
- **Mask-RCNN/**: Contains training and evaluation code, as well as scripts for running multiple tests sequentially.
- **U-Net/**: Includes training and evaluation scripts, as well as scripts for running multiple tests sequentially.
- **YOLOv8/**: Includes training and evaluation code, as well as scripts for running multiple tests sequentially.
- **Testdata/**: Contains the test datasets.
- **Trainingsdata/**: Contains the training datasets.

## License

This project is licensed under the GNU General Public License v3.0.

## Usage

To use the scripts for data augmentation, navigate to the `augmentedDataScripts` folder and run the desired script with the appropriate parameters. Similarly, processing specific training data can be done in the `customDataScripts` folder.

Training and evaluation of models can be performed in the respective model folders (`Mask-RCNN`, `U-Net`, `YOLOv8`). Each folder contains detailed instructions on how to run the scripts and set up the environment.

## Contributing

Contributions to this project are welcome. Please refer to the contributing guidelines for more information on how to submit pull requests.

## Support

If you encounter any issues or have questions about the repository, please open an issue on GitHub.
