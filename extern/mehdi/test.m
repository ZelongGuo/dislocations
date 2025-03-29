clear all; 
clc;
close all;
X = -1/3;
Y = -1/3;
Z = 16;
P1               = [-1.5; 1; -1  ];
P2               = [-1; -1.6; 1  ];
P3               = [-5; -5; -4.3 ];
nu              = 0.25;
Ss              = 1;
Ds              = -1;
Ts              = 2;
lambda = 3.3e10;
mu = 3.3e10;

% ue1 = [];
% un1 = [];
% uv1 = [];
% ue2 = [];
% un2 = [];
% uv2 = [];
% Stress1 = [];
% Strain1 = [];
% Stress2 = [];
% Strain2 = [];

[ue1,un1,uv1]=TDdispHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,nu)
[ue2,un2,uv2]=TDdispFS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,nu)
[Stress1,Strain1]=TDstressFS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,mu,lambda)
[Stress2,Strain2]=TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,mu,lambda)
