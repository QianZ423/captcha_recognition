# captcha_recognition
Captcha recognition based on deep learning

In this project, a CNN will be built through tensorflow to recognize captcha, and a simple GUI interface will be made through pyqt5 for visualization.
## Dependencies
* captcha 0.3
* tensorflow-1.15.0
* numpy 1.16.4
* PyQt5 5.15.0
* tqdm 4.28.1
* matplotlib 2.2.2
## Attention
1. Captcha is a library written in Python to generate verification codes. It supports picture verification codes and voice verification codes. I use its function to generate picture verification codes. The format of the generated verification code is 4 characters composed of 0~9 numbers and upper and lower case English letters.
2. In My code, three deep neural networks, DNN, CNN and CNN plus, are implemented to identify verification codes, and support the training of different models for final prediction.The network structure of CNN plus is as follows：or you can enter /model/imgs and find cnn.png
![cnn_plus_structure](https://github.com/QianZ423/captcha_recognition/tree/main/model/imgs/cnn.png) 

4. I use pyqt5 to make a simple GUI interface for visualizing the results of verification code generation and recognition.
5. My code has detailed Chinese comments for easy reading and modification.
## What can you learn
1. Using tensorflow to build a neural network for model training, model visualization, model preservation and recovery, and model prediction.
2. Making GUI with pyqt5.
## How to use
* First of all, you should have installed Python version 3.6 and pip.
```
pip install captcha tensorflow numpy PyQt5 tqdm matplotlib 
```
* Secondly, execute the following command:
```
cd captcha_recognition\GUI\
python main.py
```
* Finally，enter 4 characters of the verification code you want to generate in the text box at the lower left corner of the GUI interface and click the "generate verification code" button to see the generated verification code picture. Then click the "start prediction" button and wait 2 seconds to see the recognition result in the blank text box on the right.

* If you want to retrain the model, find main_ cnn.py file and execute it. If you want to evaluate the model, find eval.py file and execute it!
## Contributing
* if you want to contribute to codes, create pull request.
* if you find any bugs or error, create an issue.
## Reference
* https://github.com/ypwhs/captcha_break
## License
This project is licensed under the MIT License.
