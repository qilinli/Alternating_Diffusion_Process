# Alternating_Diffusion_Process

By [Qilin Li](https://scholar.google.com/citations?user=KpqM4F4AAAAJ&hl=en), Senjian An, Wanquan Liu, Ling Li

Curtin University.

### Introduction
This is an matlab implementation of the Alternating_Diffusion_Process (ADP). ADP is essentially an affinity learning method embeded in the
graph-based semi-supervised learning (GSSL). ADP integrates the affinity learning and label propagation of GSSL into a feedback framework,
in which the predicted pseudo labels can be used to supervise the affinity learning process.

### Usage
There are two demo test cases, simply run
```matlab
testCase_COIL20.m
testCase_synthetic_fiveCircles.m
```

### Reference

	@article{li2019semi,
  		title={Semi-supervised Learning on Graph with an Alternating Diffusion Process},
  		author={Li, Qilin and An, Senjian and Li, Ling and Liu, Wanquan},
  		journal={arXiv preprint arXiv:1902.06105},
  		year={2019}
	}


