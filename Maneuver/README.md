# This code is an extention of a flight-trained hybrid controller. Transer learning is used to train the controller to learn to maneuver around obstacles.


#### Code Structure

- `Training`: Code to train the controller to maneuver. (Python3)

#### Training

- The code has been tested on `Ubuntu 16.04+`.

- The code relies on a modified version of OpenAI baselines in `git@github.com:eanswer/openai_baselines.git` which is installed in this repo.
  ```
- Maneuver training is carried out on a pre-trained hybrid flight controller. To watch the pretrained controller fly use the following command:

  ```
  python save.py --controller model_afterIter_50
  ```
