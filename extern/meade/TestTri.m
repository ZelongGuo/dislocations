% A basic TDE example
clear;
clc

% set up the parameters of strike, slip and tensile
pr              = 0.25;
ss              = 1;
ds              = -1;
ts              = 2;

lambda = 3.3e10;
mu = 3.3e10;
% Set the observation coordinates and TDE geometry
% sx = -1;
% sy = -1;
% sz = 0;

sx = [-1/3, -1/3, -1/3, 7, -7, -1, -1, 3, -3, -1, -1, 1, -1, -1, 1];
sy = [-1/3, -1/3, -1/3, -1, -1, -3, 3, -3, 3, -1, 1, -1, -1, 1, -1];
sz = [-3,  -14/3, -6,   -5, -5, -6, -3, -6, -3, -1, -1, -1, -8, -8, -8];

% sx = [-1/3, -1/3, -1/3, -1/3 ];
% sy = [-1/3, -1/3, -1/3, -1/3 ];
% sz = [0, -3,    -5, -8 ];


X               = [-1; 1; -1;   -1;  1; -1];
Y               = [-1; -1; 1;   -1; -1; 1];
Z               = [-5; -5; -4;  -5; -5; 0];


% X               = [-1; 1; -1;   -1;  -1; 1];
% Y               = [-1; -1; 1;   -1;   1; -1];
% Z               = [-5; -5; -4;  -5;   0; -5];


%% mehdi p1-p3 counter clockwise
% X               = [-1; 1; -1;   ];
% Y               = [-1; -1; 1;   ];
% Z               = [-5; -5; -4;  ];

%% clockwise
% X               = [-1; -1; 1;   ];
% Y               = [-1; 1; -1;   ];
% Z               = [-5; -4; -5;  ];

%% Mehdi firstly EFCS --》 TDCS， check his codes figure out how it is converted，and test the TDCS with Meade's codes！！！
X2              = X;
Y2              = Y;
Z2              = Z;
% Y2              = [Y(1); Y(2); Y(3)];
% Z2              = [Z(1); Z(2); Z(3)];
% Calculate displacements at surface
U               = CalcTriDisps(sx(:), -sy(:), -sz(:), X2, -Y2, -Z2, pr, ss, -ts, -ds)
fprintf("U.x: %f\n", U.x);
fprintf("\n");
fprintf("U.y: %f\n", -U.y);
fprintf("\n");
fprintf("U.z: %f\n", -U.z);
fprintf("\n");


E               = CalcTriStrains(sx(:), -sy(:), -sz(:), X2, -Y2, -Z2, pr, ss, -ts, -ds)
% fprintf("E.xx: %f\n", E.xx);
% fprintf("\n");
% fprintf("E.yy: %f\n", E.yy);
% fprintf("\n");
% fprintf("E.zz: %f\n", E.zz);
% fprintf("\n");
% fprintf("E.xy: %f\n", -E.xy);
% fprintf("\n");
% fprintf("E.xz: %f\n", -E.xz);
% fprintf("\n");
% fprintf("E.yz: %f\n", E.yz);
% fprintf("\n");
S = StrainToStress(E, lambda, mu);
fprintf("S.xx: %f\n", S.xx);
fprintf("\n");
fprintf("S.yy: %f\n", S.yy);
fprintf("\n");
fprintf("S.zz: %f\n", S.zz);
fprintf("\n");
fprintf("S.xy: %f\n", -S.xy);
fprintf("\n");
fprintf("S.xz: %f\n", -S.xz);
fprintf("\n");
fprintf("S.yz: %f\n", S.yz);
fprintf("\n");

% from mehdi
P1 = [X(1); Y(1); Z(1)];
P2 = [X(2); Y(2); Z(2)];
P3 = [X(3); Y(3); Z(3)];
% [ue1,un1,uv1] = TDdispHS(sx(:), sy(:), sz(:), P1,P2,P3,ss,ds,ts,pr)
[ue1,un1,uv1] = TDdispHS(sx(:), sy(:), sz(:), P1,P2,P3,ss,ds,ts,pr);

% [Stress1,Strain1]=TDstressHS(sx(:),sy(:),sz(:),P1,P2,P3,ss,ds,ts,mu,lambda)
[Stress1,Strain1]=TDstressHS(sx(:),sy(:),sz(:),P1,P2,P3,ss,ds,ts,mu,lambda);
Strain1;
Stress1;

P4 = [X(4); Y(4); Z(4)];
P5 = [X(5); Y(5); Z(5)];
P6 = [X(6); Y(6); Z(6)];
[ue2,un2,uv2] = TDdispHS(sx(:), sy(:), sz(:), P4,P5,P6,ss,ds,ts,pr);
[Stress2,Strain2]=TDstressHS(sx(:),sy(:),sz(:),P4,P5,P6,ss,ds,ts,mu,lambda);
ue1 + ue2
un1 + un2
uv1 + uv2
Strain1 + Strain2;
Stress1 + Stress2



