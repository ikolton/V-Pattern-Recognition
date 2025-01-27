# V-Pattern-Recognition

*Oryginalny paper* : https://arxiv.org/pdf/2312.14135
*github* : https://github.com/penghao-wu/vstar

## Installation
To be able to use the model install dependencies from requirements.txt (it was run on windows). Requirements.txt won't install tensorrt and torch2trt, it has to be installed manually.

## Usage
When you set up environment use app.py file to run the application and select one of examples.

## Integration with nanoowl
Integration was primarily implemented in visual_search.py and VisualSearch/model/VSM.py.
You might face type errors from torch2trt and pytorch, unfortunately it requires code changes in their files.