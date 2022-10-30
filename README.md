# Facebook Marketplace Recommendation Ranking System
This project works on building and closely replicating a product ranking system of Facebook Marketplace to provide users with the most relevant products based on their search query. This is done via a multimodal pre-trained deep neural network (Text model + Image model). The model is a mini-implementation of a larger system that Facebook developed to generate product recommendation rankings for buyers on Facebook Marketplace. Shown below is a flowchart describing the overview of the system encompassing various technologies:
<img  src="https://user-images.githubusercontent.com/51030860/178149528-8a7c5b0c-3f14-46b0-b708-ff3faf455755.png">

## Table of Contents
* [Technologies Used](#technologies-used)
* [Data Cleaning](#data-cleaning)
* [Image Cleaning Operations](#image-cleaning-operations)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## Technologies Used:
This project is primarily coded using Python. The specific libraries used are listed below:
- PIL
- Pillow
- fastapi
- numpy
- os
- pandas
- pickle
- python-multipart
- requests
- scikit_image
- sklearn
- torch, torchtext, torchvision
- tqdm
- transformers
- transformers
- uvicorn

## Data Cleaning:

Before starting anything to do with deep learning, it is important to clean the data. For this project there are two datasets: [Products](Products.csv) and [Images](Images.csv). These folders contain information about the products (id, price, location, description, etc) and images (id, image id, create time, etc).

- Pandas is used to load and analyse the datasets. After exploring the datasets; it was decided to perform the following actions; convert values in price column to float, remove the column 'Unnamed 0', only keep the first part of the columns 'price_name' and 'price_description', and finally create categories from the 'category' column. Lastly the two datasets were merged into a single dataset called [Image+Products](Image+Products.csv).

- All the cleaning steps mentioned above can be found in the script [clean_tabular_data](clean_tabular_data.py).


## Image Cleaning Operations:

 After cleaning the datasets, another script [clean_images](clean_images.py) was created. Since the raw images come in a variety of sizes and aspect ratios, they need to be cleaned beforehand. Images here are modified through the PIL import & glob is used to iterate through the files within our image folder.

- Images are formed in a for loop with an enumerate function(this makes for easy naming of the photos).

- Firstly our loop opens the image and created a black background at the specified limit of 512 x 512 pixels to be overlayed later.

- the maximum dimension of each image is found and compared to our maximum acceptable size, this is computed into a ratio factor which will transform the image to the correct size.

- Background image is overlayed with the product image, image is centred on this background & saved with the enumerate function from before.


## Creating Simple Machine Learning Models:

To got a hands on with machine learning, some experimenting was done. The sklearn library was used to create two simple models: [linear_regression](linear_regression.py) and [classification_model](classification_model.py).

### Linear Regression:

- Basic attempt at inferring price using a generic linear regression model with product categories as dummy variables.
- Given the simplistic nature of the model the results reflect limited, if any, explanatory power

### Image Classification Model:

- Uses sklearn's SVC (support vector classifier) to check whether images, stored as numpy arrays, can help predict the product category it should be classified as.
- This model only produces an accuracy of 15-20%. 



## Creating the Vision Model 

Using a simpler ML model like a logistic regression is inefficient for training on an image dataset, in order to train a model for classifying images a convolutional neural network is required. This CNN utilises progressiveley more complex hidden layers to detect patterns in classes of images. This model can be found in the file [vision_model](vision_model.py).

- To start, a Pytorch dataset is created. This takes our previously cleaned images from a directory, converts them to tensors, sets up features & labels and introduces a transforms function which we will use to normalise pixel values. Then the dataset is wrapped in a dataloader where both `__getitem__` and `__len__` methods have to be incuded and this can then be fed to the model.

- Resnet-50 is imported as the CNN to be used. The first 47 layers in this model are kept frozen, the final three layers are unfrozen and additional layers are added to change RESNET-50's output layer of 2048 classes to the 13 needed for our data. As the model trains during back propergation all these layers will be adjusted to increase accuracy of predictions. A dropout layer is also used , this is  to help with regularisation.

- The model is trained through a series of 50 epochs. The accuray and loss is recorded for both training and validation cycles after each epoch. The data is saved with tensorboard, so it can be used for visualisations later on.

- After 50 epochs a final loss and final accuracy for both training & validation is taken, for this CNN the following results were obtained:

<img width="450" height="300"  src="https://user-images.githubusercontent.com/92804317/183415570-3100afd5-c85b-4be8-9fa8-7c8dee22b817.png">
<img width="450" height="300"  src="https://user-images.githubusercontent.com/92804317/183415611-ce410265-2737-4ab4-a4d7-c801afba7c0e.png">
<img width="450" height="300"  src="https://user-images.githubusercontent.com/92804317/183415647-fe2d082c-0f47-43c1-a479-b54ba5ec110c.png">

## Creating the Text model
Much like the previous CNN, this model has a similar structure. However, before creating a Pytorch dataloader text embeddings will be made using Bidirectional Encoder Representations from Transformer (BERT). This model can be found in the file [text_model](text_model.py).

- Once the correct embeddings have been loaded in, the dataloader can be made. Each sequence of text is padded with zeroes, this is to make sure all the text description have the same dimensions. The output of the dataloader will be a tensor of the description and the category ascociated with it.

- To train the text model a similar setup as the vision model is used. However, an aditional Embedding layer is required at the begininning of the model. This is a built in torch.nn function & requires the no. of words in the vocab and the embedding size (size of each sentence vector).

- The model was trained for 50 epochs and the following results were obtained:

<img width="700" height="300"  src="https://user-images.githubusercontent.com/92804317/183417296-d9befabc-7bb6-4653-a248-160570d23a04.png">
<img width="700" height="300"  src="https://user-images.githubusercontent.com/92804317/183417348-7eea7ac1-4b7b-490a-8844-a59987a369a1.png">
<img width="450"  src="https://user-images.githubusercontent.com/92804317/183417656-df4fb871-fd01-43d0-87af-f9c19da1e3e2.png">


## Combining the Models:

This script attempts to draw on the explanatory power impounded by both the prediction model using images to predict product category as well as the prediction model using text to predict product category. The script for the combined model is stored in the [combined_model](combined_model.py) file.
- Once again, just like the previous models a dataloader is utilised to complete the appropriate transformations to both text and images. However, in this model we also add an encoder & decoder to the loader. This will translate the output to the desired category name.

- Model architecture stays largely the same with one important change, the final layer needs to be adjusted so the output of both previous models has the same dimensions. After combining both models, one final linear layer is utilised to equate the final output to the number of classes.

- The combined model is very computationally expensive and takes a long time to train. Training for only 10 epochs tooks 20+ hours and even though the prediction would get better after training it longer, the model reached an accuracy of around 60% which is sufficient for the purposes of this project.

- The graph below compares the performance of all 3 models. The orange, blue and red lines show the text, image and combined models performance respectively:

<img width="700" height="300"  src="https://user-images.githubusercontent.com/110851436/198891886-9f5fe144-8874-4800-94c7-613f3194ba45.png">
<img width="700" height="300"  src="https://user-images.githubusercontent.com/110851436/198891898-a716c5d7-0baa-484c-9797-5dad6e4893bc.png">


## Containerization of the Model:
Finally in order for the model to be deployed to AWS, the model was containerized using the docker specifications in the [Dockerfile](Dockerfile), [requirements](requirements.txt) and a docker compose file [docker-compose](docker-compose.yml) constructed for convenience.

- In order to facilitate containerization, the fastapi library was used to enable end users using the model on their local machines to test how any image and its corresponding product description would be classified given the weights of the finalized combined model.

- The [api](api.py) requires 2 new dataloaders as the end user will be feeding the model with a singular image and text input, so modified loaders are used to account for this instead of the batches previously used. The new [image_processor](image_processor.py) has an added dimension for the batch size and the [text_processor](text_processor.py) now only tokenizes a single description rather than a list of descriptions.

- Text, Image & Combined model is imported and the `forward` method is called by the API `.post` method, this will ask the user to input of an image and a text description. JSONResponse is used to give the predicted output.


## AWS migration :
The last step is to migrate the containerized model to EC2 on AWS.

- Simply transfer the all needed files alongside the [api](api.py) file to the created EC2 instance and run this command `sudo cp ~/api/api.py ~/app/api.py`
- The docker image is then deployed using `docker-compose -d --env-file .env up`
- The image below shows the api running on the EC2 instance.

<img  src="https://user-images.githubusercontent.com/110851436/198884589-a367d366-c0fc-48a9-8fb9-940b80717401.png">
