## SPIN Road Mapper: Extracting Roads from Aerial Images via Spatial and Interaction Space Graph Reasoning for Autonomous Driving
[Wele Gedara Chaminda Bandara](https://www.linkedin.com/in/chamindabandara/), [Jeya Maria Jose Valanarasu](https://jeya-maria-jose.github.io/research/), and [Vishal M. Patel](https://engineering.jhu.edu/vpatel36/sciencex_teams/vishalpatel/)

Under review at IEEE International Conference on Robotics and Automation (ICRA), 2022.

## Overview of proposed SPIN module

We build graphs in two spaces: (a) spatial space and (b) a projected latent interaction space from feature maps. Graph reasoning in spatial space extracts connectivity between the road segments, whereas reasoning over interaction space delineates roads from other topographies. Nodes connected with lines in (a) denote how road segments are modeled to understand connectivity in the spatial space. Regions marked with different colors in (b) denote how different semantics are segregated for better road delineation in the interaction space.

<p align="center">
<img src="images/ICRA-intro_fig.jpeg" width="600"/>

## Architecture of proposed SPIN module and SPIN pyramid
  
The architecture of our proposed method. (a) We perform graph reasoning in both spatial and interaction space. (b) The proposed SPIN pyramid module which performs SPIN graph reasoning at multiple scales 1, 1/2, and 1/4 of original feature map to extract multi-scale long-range contextual information.

<p align="center">
<img src="images/ICCV_21-Hybrid_GR_v1.jpeg" width="600"/>
  
  
## Proposed network for road segmentation from aerial images
  
The input images are first feed forwarded to a feature extractor block followed by a bottleneck consisting of stack of two hourglass modules. Then, the output of bottleneck is passed through a segmentation branch which consists of conv layers, our SPIN pyramid and a final classification layer to get the road segmentation map.
<p align="center">
<img src="images/ICCV_21-SPIN_v1.jpeg" width="600"/>

  
## A qualitative comparison between our SPIN Road Mapper and the SOTA methods
<p align="center">
<img src="images/ICCV_21-qualitative.jpg" width="600"/>

## Reproducing the results
### 1. Donwloading the datasets
