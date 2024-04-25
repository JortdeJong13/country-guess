# country-guess
This repository holds an app where one can draw a country shape and a CNN will guess what country has been drawn.

The app can be run with python app.py
The user can draw a country and let the model guess which country has been drawn. The user can then provide feedback by selecting the drawn country from a drop down list. The user drawn countries are stored and used for model evaluation. The model is build on the [MLX](https://ml-explore.github.io/mlx/build/html/index.html) framework, as such, the app is limited to Apple Silicon.

Two jupyter notebooks are present. [Manipulate data](Manipulate data.ipynb) can used to change the reference data or the user drawings, the [Model training](Model training.ipynb) can be used to train the model.

The model is a basic CNN that produces an embedding for each drawing. Training data is generated on the fly by random augmenting the reference country. The reference country is simplified, smoothed and a series of geometric augmentations are applied. Small polygons are also removed at random. The model is trained using a triplet loss. 
