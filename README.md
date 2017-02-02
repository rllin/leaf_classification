# Leaf Classification from Kaggle

## Contents

* **Project Titles** as a level-1 heading
  - with descriptive tagline: I should be informed and intrigued. Examples:
    - "Sinatra is a DSL for quickly creating web applications in Ruby with minimal
effort"
    - "Rails is a web-application framework that includes everything needed to create
database-backed web applications according to the Model-View-Controller (MVC) pattern."
    - "Resque (pronounced like "rescue") is a Redis-backed library for creating
background jobs, placing those jobs on multiple queues, and processing
them later."

* **Overview**
  - Mixed model of 2 layer convolutional neural net trained on images and 1 layer convolutional neural net trained on features.
  - Images standardized to 128 x 128 with proportional scaling up to largest dimension of all the images (1706 x 1706) and then scaled back down

* **Getting Started**
  - Install nvidia driver according to:
    - https://github.com/NVIDIA/nvidia-docker/wiki/Deploy-on-Amazon-EC2
    - In order for it to survive restarts of your ec2 instance, refer to:
      - https://github.com/NVIDIA/nvidia-docker/issues/137
    - Use either Docker image
      - `docker pull rllin/gpu-tensorflow-python`
      - `sudo nvidia-docker run -itd --name=leaf -e "PASSWORD=password" -p 8754:8888 -p 6006:6006 rllin/gpu-tensorflow-python`
      - `sudo nvidia-docker exec -it leaf bash`
    - or requirements.txt (not tested)
      - `pip install -r requirements.txt`

* **Detailed Usage**
  - `python run_specific.py` will start a training and validation session using the following hyperparamaters:
      
      ```json
      {
        "f_conv1_num": 8,
        "f_conv1_out": 512,
        "f_d_out": 1024,
        "f_dropout": 0.8255444236474426,
        "conv1_num": 5,
        "conv1_out": 128,
        "conv2_num": 7,
        "conv2_out": 256,
        "d_out": 1024,
        "dropout": 0.7296244459829335,
        "report_interval": 100,
        "l2_penalty": 0.01,
        "LEARNING_RATE": 0.001,
        "TRAIN_SIZE": 1.0,
        "WIDTH": 128,
        "SEED": 42,
        "BATCH_SIZE": 66,
        "ITERATION": 5000.0,
        "HEIGHT": 128,
        "CHANNEL": 1,
        "VALIDATION_SIZE": 0.2,
        "NUM_CLASSES": 99,
        "CLASS_SIZE": 1.0,
        'features_images': 'features only'
      }
      ```
    - `LEARNING_RATE` looks like a fixed parameter, but it is searched over also.
  - These hyperparameters will achieve fairly high validation accuracy 80% for features only run after 5000 iterations.
    - However, running the above `python run_specific.py` is meant to be images + features, and these hyperparamters don't seem to break 60% validation accuracy.

* **Next steps**
  - I've ordered next possible steps in decreasing combined ease of implementation and expected marginal benefit:
    - Consider third convolutional layer to farther pool and shrink image.
    - Combine dropout with max norm rather than l2_loss as that seems to be suggested as best for preventing exploding or imploding weights.
    - Include image sizes as hyperparamters to search over.  Most of this code is in place already.
    - Write a better batching process that's more pipe like, perhaps workers create or find images based on size based on hyperparameters of image size.

* **References**
  - https://github.com/alrojo/tensorflow-tutorial/blob/master/lab4_Kaggle/lab4_Kaggle.ipynb
  - https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network.ipynb
  - https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
  - https://github.com/fluxcapacitor/pipeline/wiki/AWS-GPU-TensorFlow-Docker
  - https://github.com/NVIDIA/nvidia-docker/issues/137

