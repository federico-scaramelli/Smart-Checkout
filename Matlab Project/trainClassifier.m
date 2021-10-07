% trainClassifier
% Richiede all'utente una cartella contenente i dati di training, calcola i
% descrittori di tali immagini. Crea un ClassificationKNN  con distanza 
% Cityblock addestrato sul featuresVectors calcolato con computeFeatures(). 
% Al termine, salva il classificatore a scelta dell'utente.

clear all
close all

dataFolder = uigetdir('./', 'Select training data folder');

%Calcolo delle features per ogni immagine segmentata
featuresVectors = computeFeatures(dataFolder);

disp('Creating classifier');

numRows = size(featuresVectors,2);

gt = table2array(featuresVectors(:, numRows)); %Groundtruth
descriptors = featuresVectors(:, 1:numRows-1); %Descrittori di ogni oggetto (features)

k = 1; 
knn = fitcknn(descriptors, gt, 'Distance', 'euclidean', ...
    'NumNeighbors', k, 'Standardize', 1);

[file,~,~] = uiputfile('.\classifier_knn.mat');
save(file, 'knn', 'featuresVectors');

disp(['Classifier ' file ' created']);