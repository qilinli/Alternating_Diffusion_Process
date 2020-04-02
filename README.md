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

If you find the code is useful, please consider to cite:

	@article{Li2018affinity,
		author = {Qilin Li and Wanquan Liu and Ling Li},
		title = {Affinity Learning via a Diffusion Process for Subspace Clustering},
		journal = {Pattern Recognition},
		year = {2018}
	}


