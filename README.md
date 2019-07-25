# sullivan_ryan
Activity tracking for Ryan, REU Summer 2019

All data used in this project is from the rib fracture dataset. At the time of writing this, that is located on the Midi Lab workstation, CookieMonster, at `/data/mididata/rib_fracture_id/`

## File descriptions

### HPCC Traininf System
* test_model.py - trains a model using the hyperparameters defined at the top of the file. Saves important training information into the Training folder, training history into the History folder, model weights into the Weights folder, and sample predictions into the Sample_Predictions folder.
* generate_predictions.py - imports the hyperparameters from the training file and weights from the weights file to save image predictions for the test set.
* view_masks.py - displays the prediction using matplotlib. Change the matplotlib backend to work over X11 or locally.

### Model
* UNet-with-ResNet50.ipynb - contains the code used to create and test the combined UNet + ResNet model. Used to generate **UNet-with-Resnet50.py**
* Resnet-Classification.ipynb - used to train the line/no line binary classification model
* gridsearch.py - randomly selects hyperparameter values from predefined ranges. Performs a random hyperparameter search (which has been show to be more effective than grid search, albeit less easy to interpret). Written to run on the HPCC.

### Scripts
* sort.py - A curses based interface for labeling binary classification data. Plots jpegs or matlab files using matplotlib. Use arrow keys to sort images into bins. Hit enter to stop, and resume from where you left off with the command stored in `resume.txt`. For jpeg images, a single image is displayed. For matfiles, 4 images of varying grayscale ranges are shown to simplify reading radiographs.

### Data
* line.txt - List of files containing lines in the Stanford CheXpert dataset
* noline.txt - List of files containing lines in the Stanford CheXpert dataset
* ped_line.txt - List of files containing lines in the rib fracture dataset (only including children ages 5 and below)
* ped_noline.txt - List of files containing lines in the rib fracture dataset (only including children ages 5 and below)
