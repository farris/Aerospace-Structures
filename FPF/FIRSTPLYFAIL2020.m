%% SETUP
tic
close all  % Close all files
clear all  % Clear all variables
clc        % Clear command line 
tic
inFile  = 'SE142_3_Laminate_Input.xlsx';  % Excel file with input data
inFile2  = 'SE142_Material_DataBase.xlsx';
outFile = 'SE142_3_Laminate_Output1.xlsx'; % Excel file for calculated output
input =  xlsread(inFile, 1, 'E43:E51');
inputu = input(1);
outputu = input(2);
n = input(3);
NX= input(4);
NY = input(5);
NXY  =input(6);
MX = input(7);
MY = input(8);
MXY = input(9);
SF = xlsread(inFile, 1, 'E54');
%% INPUT MANIPULATION
compID = xlsread(inFile, 1, 'I44:I55');
compID = flip(compID)';
rot = (xlsread(inFile, 1, 'J44:J55'))';
rot = flip(rot);
tply = xlsread(inFile, 1, 'K44:K55');
tply = flip(tply)';
[comp_all,compnames,rawc] = xlsread(inFile2, 1, 'E55:V85');

comp_p = [];
compIDloop = [1:8];
k = 1;
while true
for i = 1:8
if compID(k) == compIDloop(1,i)
     comp_p(:,k) = comp_all(3:31,i);
     compname = compnames{1,i};
end
end
k = k+1;
if k == n + 1
    break
end
end
comp_p(12:20,:) = [];
%% PROPERTY ASSIGNMENT
vf = comp_p(1,:);
E1 = comp_p(2,:);
E2 = comp_p(3,:);
E3 = comp_p(4,:);
G12 = comp_p(5,:);
G13 = comp_p(6,:);
G23 = comp_p(7,:);
v12 = comp_p(8,:);
v13 = comp_p(9,:);
v23 = comp_p(10,:);
rho = comp_p(11,:);
f1t = comp_p(12,:);
f2t = comp_p(13,:);
f3t = comp_p(14,:);
f1c = comp_p(15,:);
f2c  = comp_p(16,:);
f3c = comp_p(17,:);
fs6 = comp_p(18,:);
fs5 = comp_p(19,:);
fs4 = comp_p(20,:);
%% LAMINA STIFFNESS BEHAVIOR
for j = 1:n
v21(j) = v12(j)*(E2(j)/E1(j));
v31(j) = v13(j)*(E3(j)/E1(j));
v32(j) = v23(j)*(E3(j)/E2(j));
q11(j) = E1(j)/(1-(v12(j)*v21(j)));
q12(j) = (v12(j)*E2(j))/(1-(v12(j)*v21(j)));
q22(j) = E2(j)/(1-(v12(j)*v21(j)));
q66(j) = G12(j);

Q(:,:,j) = [q11(j) q12(j) 0;
     q12(j) q22(j) 0;
     0   0   q66(j)];

S(:,:,j) = [1/E1(j) -v12(j)/E1(j) 0;
      -v12(j)/E1(j) 1/E2(j) 0;
      0  0   1/G12(j)];
end
%% QBAR/SBAR
for i= 1:n
c(i) = cosd(rot(i));
s(i) = sind(rot(i));

T1PR(:,:,i) = [c(i)^2 s(i)^2 2*c(i)*s(i)
        s(i)^2 c(i)^2 -2*c(i)*s(i);
        -c(i)*s(i) c(i)*s(i) ((c(i)^2)-(s(i)^2))];

T2PR(:,:,i) = [c(i)^2 s(i)^2 -2*c(i)*s(i)
        s(i)^2 c(i)^2  2*c(i)*s(i);
        c(i)*s(i) -c(i)*s(i) ((c(i)^2)-(s(i)^2))];
    
QBAR(:,:,i)=  T2PR(:,:,i)*Q(:,:,i)*(T2PR(:,:,i))';
SBAR(:,:,i) = inv(QBAR(:,:,i));
end
%% A MATRIX
for i = 1:n
A(:,:,i) = QBAR(:,:,i) * tply(i);
end
A = (sum(A,3));
%% B MATRIX
%if odd%%%ply add on%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if mod(n,2) ~= 0
tply = [tply(1:(n+1)/2) tply((n+1)/2:end)];
tply((n+1)/2) = tply((n+1)/2)/2;
tply((n+1)/2 + 1) = tply((n+1)/2 + 1)/2 ;
end
%if odd%%%ply add on%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%ALWAYS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bply = length(tply)/2;
tvec = horzcat(tply(1:bply),0);
tvecc = cumsum(flip(tvec));
BDIS =  flip(-diff(tvecc.^2));
%ALWAYS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%if odd%%%qbar add on%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if mod(n,2) ~= 0
B = QBAR(:,:,(n+1)/2);             %B = Matrix add-on
l = length(QBAR);
d = (n+1)/2 ;                      %dimension where we need to insert B
QBAR(:,:,1:d-1) = QBAR(:,:,1:d-1);
QBAR(:,:,d) = B;
QBAR(:,:,d+1:l+1) = QBAR(:,:,d:l);
end
%if odd%%%qbar add on%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%ALWAYS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:bply
    bloopbottom(:,:,i) = QBAR(:,:,i)*BDIS(1,i);
end
%top plies
bply = length(tply)/2;
tvect = horzcat(0,tply(n/2+1:end));
tvecct = cumsum((tvect));

BDIStop =  diff(tvecct.^2);
topqA = QBAR(:,:,(bply+1):end);
    for i = 1:bply
    blooptop(:,:,i) = topqA(:,:,i)*BDIStop(1,i);
    
    end
B = ((sum(bloopbottom,3)+sum(blooptop,3))/2);
%ALWAYS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% D MATRIX
%ALWAYS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bply = length(tply)/2;
tvec = horzcat(tply(1:bply),0);
tvecc = cumsum(flip(tvec));
BDIS =  flip(diff(tvecc.^3));
%ALWAYS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ALWAYS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:bply
    bloopbottom(:,:,i) = QBAR(:,:,i)*BDIS(1,i);
end
%top plies
bply = length(tply)/2;
tvect = horzcat(0,tply(n/2+1:end));
tvecct = cumsum((tvect));
BDIStop =  diff(tvecct.^3);
topqA = QBAR(:,:,(bply+1):end);
    for i = 1:bply
    blooptop(:,:,i) = topqA(:,:,i)*BDIStop(1,i);
    
    end
D = ((sum(bloopbottom,3)+sum(blooptop,3))/3);
%ALWAYS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initiliazing ABD
ABD = ones(6,6);
ABD(1:3,1:3) = A;
ABD(1:3,4:6) = B;
ABD(4:6,1:3) = B;
ABD(4:6,4:6) = D;
ABDinv = (inv(ABD));
%% Equivalent laminate stiffness properties  
tlam = sum(tply)*n;
EX = det(A)/(((A(2,2)*A(3,3)) - (A(2,3)^2))*tlam);
EY = det(A)/(((A(1,1)*A(3,3)) - (A(1,3)^2))*tlam);
GXY = det(A)/(((A(1,1)*A(2,2)) - (A(1,2)^2))*tlam);
vxy = - (((A(1,3)*A(2,3)) - (A(1,2)*A(3,3))) / ((A(2,2)*A(3,3)) - (A(2,3)^2)));
nxxy = (((A(1,2)*A(2,3)) - (A(1,3)*A(2,2))) / ((A(2,2)*A(3,3)) - (A(2,3)^2)));
nyxy = (((A(1,2)*A(1,3)) - (A(1,1)*A(2,3))) / ((A(1,1)*A(3,3)) - (A(1,3)^2)));
IC = (EX+EY)/((4*GXY)*(1+vxy));
%% APPLIED LOAD ANALYSIS
stressvec = [NX NY NXY]';
momvec = [MX MY MXY]';
vvs = vertcat(stressvec, momvec);
vvs1 = ABDinv*vvs;
lamstrain0 = vvs1(1:3);
lamkurv0 = vvs1(4:6);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bplycord = [flip(tvecct) -tvecc(2:end)]';
coords = repelem(bplycord,2);
CORD = coords(2:end-1);
CORD = flip(CORD);                          %FINAL COORDINATE SYSTEM
QBARN ={}                ;                 %converting QBAR INTO CELL 
for i = 1:length(tply)
    QBARN{i} = QBAR(:,:,i);
end
QBARN = repelem(QBARN,2);
%% STRUCTURAL
%STRAINS
for i = 1: length(CORD)
lamstrainf(:,:,i) = lamstrain0 + (CORD(i)*lamkurv0);
lamstrains{i} = lamstrainf(:,:,i);
end
%STRESSES
for i = 1:length(QBARN)
sigmas{i} = QBARN{i} * lamstrains{i};
end
%% LOCAL
if mod(n,2) ~= 0
BUSS = rot((n+1)/2)  ;                %ROT MATRIX ODD PLY ADD ON
l = length(rot);
d = (n+1)/2 ;                      
rot(1,1:d-1) = rot(1,1:d-1);
rot(1,d) = BUSS;
rot(1,d+1:l+1) = rot(1,d:l);
end
rot = repelem(rot,2);
for i= 1:length(rot)
c(i) = cosd(rot(i));
s(i) = sind(rot(i));
T1P(:,:,i) = [c(i)^2 s(i)^2 2*c(i)*s(i)
        s(i)^2 c(i)^2 -2*c(i)*s(i);
        -c(i)*s(i) c(i)*s(i) ((c(i)^2)-(s(i)^2))];
T2P(:,:,i) = [c(i)^2 s(i)^2 -2*c(i)*s(i)
        s(i)^2 c(i)^2  2*c(i)*s(i);
        c(i)*s(i) -c(i)*s(i) ((c(i)^2)-(s(i)^2))];
T1PRF{i} = T1P(:,:,i);
T2PRF{i} = T2P(:,:,i);
end
for i= 1:length(rot)
sigmalocal{i} = (T1PRF{i} * sigmas{i});
end
%% MARGINS OF SAFETIES
if mod(n,2) ~= 0
f1t = [f1t(1:(n+1)/2) f1t((n+1)/2:end)];
f1c = [f1c(1:(n+1)/2) f1c((n+1)/2:end)];
f2t = [f2t(1:(n+1)/2) f2t((n+1)/2:end)];
f2c = [f2c(1:(n+1)/2) f2c((n+1)/2:end)];
fs6 = [fs6(1:(n+1)/2) fs6((n+1)/2:end)];
end
for i = 1:length(tply)
F1T(i) = f1t(i)/SF;
F1C(i) = f1c(i)/SF;
F2T(i) = f2t(i)/SF;
F2C(i) = f2c(i)/SF;
FSS(i) = fs6(i)/SF;
end
%double the factors to match number of top and bottom
F1T = repelem(F1T,2);
F1C = repelem(F1C,2);
F2T = repelem(F2T,2);
F2C = repelem(F2C,2);
FSS = repelem(FSS,2);

for i = 1:length(F1T)
   
    %MOS 1
if (sigmalocal{i}(1))/abs(sigmalocal{i}(1)) == 1
    ms1(i) = F1T(i)/(sigmalocal{i}(1)) -1 ;
else
    ms1(i) = F1C(i)/(sigmalocal{i}(1)) -1 ;
end

    %MOS2
if (sigmalocal{i}(2))/abs(sigmalocal{i}(2)) == 1
    ms2(i) = F2T(i)/(sigmalocal{i}(2)) -1;
else
    ms2(i) = F2C(i)/(sigmalocal{i}(2)) -1;
end

    %SHEAR
    mss(i) = (FSS(i)/(abs(sigmalocal{i}(3)))) -1;
    
    %Combining
MOS(i,:) = [ms1(i) ms2(i) mss(i)];
end  
%%%%%%%%%%%%if odd
if mod(n,2) ~= 0 
MOS(length(ms1)/2:(length(ms1)/2) + 1,:)= [];
end  

minimum = min(min(MOS));
[r,c] = find(MOS == minimum);
if mod(r(1),2) ~= 0 
plyfail = (r(1)+1)/2;
else
plyfail = r(1)/2;
end
%% FINAL RESULTS
if c ==1 
    disp('Fiber Fail in ply number:')
    disp(plyfail)
    disp('MOS OF PLY:')
    disp(minimum)
elseif c == 2 
    disp('Matrix Fail in ply number:')
    disp(plyfail)
    disp('MOS OF PLY:')
    disp(minimum)
elseif c == 3
    disp('Shear Fail in ply number:')
    disp(plyfail)
    disp('MOS OF PLY:')
    disp(minimum)
end
toc








