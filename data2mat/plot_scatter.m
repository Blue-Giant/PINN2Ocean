clear all
clc
close all
data2points = load('testData2XY.mat');
points2XY = double(data2points.Points2XY)';
figure('name','domian')
scatter(points2XY(1,:),points2XY(2,:),'r.')
hold on

data2solus=load('test_solus.mat');
solu2true = double(data2solus.Utrue);
solu2dnn = double(data2solus.UsinADDcos);

figure('name','solus2dnn')
cp = solu2dnn;
scatter3(points2XY(1,:),points2XY(2,:),solu2dnn,50,cp,'.');
grid on
colorbar
caxis([0,1e-1])
hold on

figure('name','solus2true')
cp2true = solu2true;
scatter3(points2XY(1,:),points2XY(2,:),solu2true,50,cp2true,'.');
grid on
colorbar
caxis([0,1e-1])
hold on

