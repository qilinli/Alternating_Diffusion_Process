function W = weight_b_g_ag(X, K, sigma_k, type)

%%% Edge weight functions %%%
%%% Input:
%%%       X: Nxd data points
%%%       K: number of neighbors
%%%       type(optional): 1(binary), 2(gaussian), 3(adaptive gaussian)
%%% Output:
%%%        W: NxN weight matrix
%%%
%%%  Copyrights @ QILIN LI, 11/04/2018

if type == [], type = 2; end  % default gaussian 

connectionoptions.KernelType='Linear';    
connectionoptions.Display=0;
connectionoptions.KB=K;    
connectionoptions.Model='KNN';
[P, Pnn]=ConnectionModel(X,connectionoptions);

weightoptions.Type=type;
weightoptions.KB=K;
weightoptions.Display=0;
weightoptions.KernelSize=cal_sigma(X, sigma_k);
%weightoptions.KernelSize=0.3;
W=WeightingModel(X,P,Pnn,weightoptions);
W=full(W);