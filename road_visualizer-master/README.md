# Road visualizer for Spacenet and DeepGlobe dataset

This software is a modification of Visualizer from Spacenet competition for road extraction. I made it work for DeepGlobe dataset also. The original competition and code are at https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=17036&pm=14735.

## Usage
For Spacenet dataset:
```
java -jar visualizer.jar -params ./data/params.txt
```

For DeepGlobe dataset:
```
java -jar visualizerDG.jar -params ./data/paramsDG.txt
```
Please change the corresponding parameters in `paramsDG.txt` to the image folder, ground truth, and mask prediction.
The following image shows a screenshot of a APLS visualization:

![APLS visualizer](visualizer_example.png)
