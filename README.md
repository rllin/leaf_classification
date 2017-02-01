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
  - installation & prerequisites
  - how to run examples and tests
    - include a `Procfile` to start any necessary servers or daemon processes

* **Detailed Usage**
  - models and interface
  - examples
  - configuration
  - middleware or plugins
  - how it works

* **References**
  - https://github.com/alrojo/tensorflow-tutorial/blob/master/lab4_Kaggle/lab4_Kaggle.ipynb
  - https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network.ipynb
  - https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
  - https://github.com/fluxcapacitor/pipeline/wiki/AWS-GPU-TensorFlow-Docker
  - https://github.com/NVIDIA/nvidia-docker/issues/137



*
## Formatting

* Call the file `README.md`.
* Write in markdown format.
  - You should use triple backtick blocks for code, and supply a language prefix:

        ```ruby
        def hello(str)
          puts "hello #{str}!"
        end
        ```

* Do not wrap lines. In emacs, enable the `longlines-mode` to make your document word wrap intelligently.


## Supporting Documentation

Besides a `README.md`, your repo should contain a `CHANGELOG.md` summarizing major code changes, a `LICENSE.md` describing the code's license (typically Apache 2.0 for our open-source projects, All Rights Reserved for internal projects), and a `notes/` directory that is a git submodule of the project's wiki. See the [style guide for repo organization](https://github.com/infochimps-labs/style_guide/blob/master/style-guide-for-repo-organization.md) for more details.

