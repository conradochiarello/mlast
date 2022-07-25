clc

trainRatio = 0.40;
valRatio = 0.05;

numberOfInputs = 9;
trainingFunction = 'trainbr';
numberOfLayers = 2;
physicsModel = 'st';

M = 28.97;
R = 8314;
g = 9.81;
T = 298;

testRatio = 1 - trainRatio - valRatio;

dataSets = fieldnames(Resultados);
pts = "P" + (1:18);

ms = 1:5;

nd = numel(dataSets);
np = numel(pts);
nm = numel(ms);

ni = nd*(np-1)*(sum(1 : nm - 1));

% inputs
% (parâmetros) slug tracking, condições experimentais (LS, LB, RGB, VB, ...)

% outputs
% (parâmetros) experimental

% dados

% autor > bateria > ponto > estação > medidas

% entradas e saídas slug tracking

SM = zeros(ni, 3);
PM = zeros(ni, numberOfInputs - 1);
HM = zeros(ni, numberOfInputs);
dP = zeros(ni, 1);
dx = zeros(ni, 1);
LB = zeros(ni, 1);
LS = zeros(ni, 1);
VB = zeros(ni, 1);
f = zeros(ni, 1);
RGB = zeros(ni, 1);

JL = [0.7, 0.5, 0.3, 1.0, 0.5, 1.5, 0.75, 1.0, 1.3, 0.7, 0.5, 2.0, 1.5, 1.0, 3.0, 2.5, 2.0, 1.5];

ns = 0;
i = 0;

for d = 1 : nd
    for p = 1 : np
        
        if p == 9
            continue
        end
        
        for m1 = 1 : nm - 1
            for m2 = m1 + 1 : nm
                clc
                
                i = i + 1;
                ns = ns + 1;
                
                [P1e, x1e, ~, ~, ~, ~, ~, ~, ~] = getData(p, m1, "E");
                [P2e, x2e, ~, LB2e, LS2e, ~, VB2e, f2e, RG2e] = getData(p, m2, "E");
                
                dx21 = x2e - x1e;
                
                dP_msr = P1e - P2e;
                
                if dP_msr < 0
                    keyboard
                end
                
                rho_L = 999;
                rho_G = (P2e + 1013.25)*M/(R*T);
                mu_L = 1.0016E-3;
                mu_G = 1.81E-5;
                J_L = JL(p);
                
                [P1s, ~, ~, ~, ~, ~, ~, ~, ~] = getData(p, m1, "S");
                [P2s, ~, ~, LB2s, LS2s, JG2s, VB2s, f2s, RG2s] = getData(p, m2, "S");
                
                dP_mdl = P1s - P2s;
                
                J_G = JG2s;
                
                D = 26e-3;
                angle = 0;
                
                Re_SL = rho_L*J_L*D/mu_L;
                Re_SG = rho_G*J_G*D/mu_G;
                
                HM(i, :) = [dP_mdl, LB2s, LS2s, VB2s, f2s, RG2s, Re_SL, Re_SG, dx21];
                dP(i) = dP_msr;
                dx(i) = x2e - x1e;
                LB(i) = LB2e;
                LS(i) = LS2e;
                VB(i) = VB2e;
                f(i) = f2e;
                RGB(i) = RG2e;
                
            end
        end
    end
end

%%

clc

inputsToHybrid = HM';
outputs = [dP, LB, LS, VB, f, RGB]';

clc
disp("Training model...")
[hybridModel, trh] = setnet(numberOfInputs, trainingFunction, inputsToHybrid, outputs, numberOfLayers, trainRatio, valRatio, testRatio);

physicalModelResults = HM(:, 1:6)';
hybridModelResults = hybridModel(inputsToHybrid);

clc
disp("Done!")

dPh = hybridModelResults(1,:);
dPp = physicalModelResults(1,:);
dPe = outputs(1,:);

LBh = hybridModelResults(2,:);
LBp = physicalModelResults(2,:);
LBe = outputs(2,:);

LSh = hybridModelResults(3,:);
LSp = physicalModelResults(3,:);
LSe = outputs(3,:);

VBh = hybridModelResults(4,:);
VBp = physicalModelResults(4,:);
VBe = outputs(4,:);

fh = hybridModelResults(5,:);
fp = physicalModelResults(5,:);
fe = outputs(5,:);

RGh = hybridModelResults(6,:);
RGp = physicalModelResults(6,:);
RGe = outputs(6,:);

% Pressure
MAPE_P_P = mape(dPp, dPe);
MAPE_H_P = mape(dPh, dPe);

% LB
MAPE_P_B = mape(LBp, LBe);
MAPE_H_B = mape(LBh, LBe);

% LS
MAPE_P_S = mape(LSp, LSe);
MAPE_H_S = mape(LSh, LSe);

% VB
MAPE_P_V = mape(VBp, VBe);
MAPE_H_V = mape(VBh, VBe);

% f
MAPE_P_F = mape(fp, fe);
MAPE_H_F = mape(fh, fe);

% RG
MAPE_P_R = mape(RGp, RGe);
MAPE_H_R = mape(RGh, RGe);

close all

% figure(1)
% 
% xlabel('Real \Delta{}P', 'FontSize', 16)
% ylabel('Predicted \Delta{}P', 'FontSize', 16)
% 
% str_physics_model = 'Slug Tracking';
% str_dataset_train='(Barros, 2019)';
% str_dataset_test='(Barros, 2019)';
% 
% train_test_portion=100*trainRatio;
% 
% str_dataset_train_all=[num2str(train_test_portion) '% ' str_dataset_test];
% 
% 
% strTitle={['Physics Model: ' str_physics_model],['Machine Learning trained with ' str_dataset_train_all],['Tested with: ' str_dataset_test]};
% title(strTitle, 'FontSize', 20)
% minlim = 0;
% maxlim = max([max(outputs), max(hybridModelResults), max(physicalModeldP)]);
% minmax = [minlim maxlim];

plotThis('\Delta{}P', dPh, dPp, dPe, MAPE_P_P, MAPE_H_P, trainRatio)
plotThis('L_B', LBh, LBp, LBe, MAPE_P_B, MAPE_H_B, trainRatio)
plotThis('L_S', LSh, LSp, LSe, MAPE_P_S, MAPE_H_S, trainRatio)
plotThis('V_B', VBh, VBp, VBe, MAPE_P_V, MAPE_H_V, trainRatio)
plotThis('f', fh, fp, fe, MAPE_P_F, MAPE_H_F, trainRatio)
plotThis('R_G', RGh, RGp, RGe, MAPE_P_R, MAPE_H_R, trainRatio)

function [net, tr] = setnet(numberOfInputs, trainingFunction, inputs, outputs, numberOfLayers, trainRatio, valRatio, testRatio)
layers = round(linspace(numberOfInputs, 6, numberOfLayers + 1));
net = cascadeforwardnet(layers, trainingFunction);
net.trainParam.showWindow = false;
% net.divideFcn = 'divideblock';
net.divideParam.trainRatio = trainRatio;
net.divideParam.valRatio = valRatio;
net.divideParam.testRatio = testRatio;


[net, tr] = train(net, inputs, outputs);
end

function [P, x, xD, LB, LS, JG, VB, f, RG] = getData(pt, st, dataFlag)
opts = delimitedTextImportOptions("NumVariables", 15);

% Specify range and delimiter
opts.DataLines = [3, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["Var1", "posm", "LD", "LBD", "LSD", "PjhPa", "Jgms", "VBms", "FHz", "RG", "Var11", "Var12", "Var13", "Var14", "Var15"];
opts.SelectedVariableNames = ["posm", "LD", "LBD", "LSD", "PjhPa", "Jgms", "VBms", "FHz", "RG"];
opts.VariableTypes = ["string", "double", "double", "double", "double", "double", "double", "double", "double", "double", "string", "string", "string", "string", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["Var1", "Var11", "Var12", "Var13", "Var14", "Var15"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["Var1", "Var11", "Var12", "Var13", "Var14", "Var15"], "EmptyFieldRule", "auto");

% Import the data
res = readtable("/Users/conradochiarello/Google Drive/MATLAB/Specrux/+data/+st/" + dataFlag + "_medias_P" + pt + ".SLUG", opts);

% Import the data

x = res.posm;
xD = res.LD;
LB = res.LBD;
LS = res.LSD;
P = res.PjhPa;
JG = res.Jgms;
VB = res.VBms;
f = res.FHz;
RG = res.RG;

if dataFlag == "S"
    realStationIndex = [12, 16, 18, 25, 32];
    
    x = x(realStationIndex);
    xD = xD(realStationIndex);
    LB = LB(realStationIndex);
    LS = LS(realStationIndex);
    P = P(realStationIndex);
    JG = JG(realStationIndex);
    VB = VB(realStationIndex);
    f = f(realStationIndex);
    RG = RG(realStationIndex);
end

x = x(st);
xD = xD(st);
LB = LB(st);
LS = LS(st);
P = P(st);
JG = JG(st);
VB = VB(st);
f = f(st);
RG = RG(st);

end

function MAPE = mape(model, experimental)
MAPE = round(100*mean(abs(model - experimental)./experimental), 2);
end

function plotThis(variable, hybrid, physical, experimental, mape_phys, mape_hyb, trainRatio)
maxlim = max([max(experimental), max(hybrid), max(physical)]);
minmax = [0 maxlim];

str_dataset_test='(Barros, 2019)';
train_test_portion=100*trainRatio;

str_dataset_train_all=[num2str(train_test_portion) '% ' str_dataset_test];

strTitle={'Physics Model: Slug tracking',['Machine Learning trained with ' str_dataset_train_all],['Tested with: ' str_dataset_test]};
        
figure
hold on
title(strTitle, 'FontSize', 20)
plot(experimental, physical, 'gx')
plot(experimental, hybrid, 'rx')
plot(minmax, minmax, 'k--')
xlim(minmax)
ylim(minmax)
legend({"Physics model MAPE: " + mape_phys + "%", "Hybrid ML MAPE: " + mape_hyb + "%"}, 'FontSize', 16, 'Location', 'southeast')
xlabel(['Real ', variable], 'FontSize', 16)
ylabel(['Predicted ' variable], 'FontSize', 16)
hold off
end