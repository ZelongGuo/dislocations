% A basic TDE example
clear;
clc

% set up the parameters of strike, slip and tensile
pr              = 0.25;
ss              = 5;
ds              = 0;
ts              = 0;

lambda = 3.3e10;
mu = 3.3e10;

% %% ----------------------------------------------------------------------
% % Set the observation coordinates and TDE geometry
% sx = [10, 10, ];
% sy = [8, 8, ];
% sz = [0, -1, ];
% X               = [-2; 2; 2;       2; 2; -2];
% Y               = [0; 0; 0;        0; 0; 0];
% Z               = [-1; -1; -5;     -1; -5; -5];
% 
% % X               = [-2; 2; 2;       2; -2; 2];
% % Y               = [0; 0; 0;        0; 0; 0];
% % Z               = [-1; -1; -5;     -1; -5; -5];
% %% ----------------------------------------------------------------------

%% ----------------------------------------------------------------------
% Set the observation coordinates and TDE geometry
sx = [0.1,   0.1, ];
sy = [0.1,   0.1, ];
sz = [-0.9, -1.1, ];
X               = [ 1; -1;  1;        -1; -1;  1];
Y               = [-1;  1;  1;        -1;  1; -1];
Z               = [-1; -1; -1;        -1; -1; -1];

%% ----------------------------------------------------------------------

%% Mehdi firstly EFCS --》 TDCS， check his codes figure out how it is converted，and test the TDCS with Meade's codes！！！
X2              = X(1:3);
Y2              = Y(1:3);
Z2              = Z(1:3);
X3              = X(4:6);
Y3              = Y(4:6);
Z3              = Z(4:6);
% Y2              = [Y(1); Y(2); Y(3)];
% Z2              = [Z(1); Z(2); Z(3)];
% Calculate displacements at surface
U1               = CalcTriDisps(sx(:), -sy(:), -sz(:), X2, -Y2, -Z2, pr, ss, -ts, -ds);
U2               = CalcTriDisps(sx(:), -sy(:), -sz(:), X3, -Y3, -Z3, pr, ss, -ts, -ds);
U.x = U1.x + U2.x
U.y = U1.y + U2.y
U.z = U1.z + U2.z
fprintf("U.x: %f\n", U.x);
fprintf("\n");
fprintf("U.y: %f\n", -U.y);
fprintf("\n");
fprintf("U.z: %f\n", -U.z);
fprintf("\n");


E1               = CalcTriStrains(sx(:), -sy(:), -sz(:), X2, -Y2, -Z2, pr, ss, -ts, -ds);
E2               = CalcTriStrains(sx(:), -sy(:), -sz(:), X3, -Y3, -Z3, pr, ss, -ts, -ds);
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
S1 = StrainToStress(E1, lambda, mu);
S2 = StrainToStress(E2, lambda, mu);
S = S1 + S2
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
P4 = [X(4); Y(4); Z(4)];
P5 = [X(5); Y(5); Z(5)];
P6 = [X(6); Y(6); Z(6)];
[ue1,un1,uv1] = TDdispHS(sx(:), sy(:), sz(:), P1,P2,P3,ss,ds,ts,pr);
[ue2,un2,uv2] = TDdispHS(sx(:), sy(:), sz(:), P4,P5,P6,ss,ds,ts,pr);
ue3 = ue1 + ue2
un3 = un1 + un2
uv3 = uv1 + uv2

% [Stress1,Strain1]=TDstressHS(sx(:),sy(:),sz(:),P1,P2,P3,ss,ds,ts,mu,lambda)
[Stress1,Strain1]=TDstressHS(sx(:),sy(:),sz(:),P1,P2,P3,ss,ds,ts,mu,lambda);
[Stress2,Strain2]=TDstressHS(sx(:),sy(:),sz(:),P4,P5,P6,ss,ds,ts,mu,lambda);
Stress1 + Stress2


% 
% [X,Y,Z] = meshgrid(-3:.02:3,-3:.02:3,-5);
% [ue,un,uv] = TDdispHS(X,Y,Z,[-1 0 0],[1 -1 -1],[0 1.5 -2],-1,2,3,.25);
% h = surf(X,Y,reshape(ue,size(X)),'edgecolor','none');
% view(2)
% axis equal
% axis tight
% set(gcf,'renderer','painters')
