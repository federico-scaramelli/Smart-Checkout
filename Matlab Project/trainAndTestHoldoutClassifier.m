% Addetsra e testa un classificatore dividendo il dataset in modalità
% holdout casuale. Usato per scoprire il modello finale di
% classificazione da utilizzare, iterando su nuovi dati e parametri.

clear all
close all

%Selezione da parte dell'utente della cartella contenente il dataset
dataFolder = uigetdir('.\', 'Select training data folder');

%Calcolo delle features per ogni immagine contenuta nella cartella
featuresVectors = computeFeatures(dataFolder);

%Creazione di due partizioni casuali del dataset
cv = cvpartition(size(featuresVectors,1), 'Holdout', 1/4);
tr = cv.training(1);
ts = cv.test(1);

training.features = featuresVectors(tr,1:size(featuresVectors,2)-1);
training.lbl = featuresVectors(tr,size(featuresVectors,2));
test.features = featuresVectors(ts,1:size(featuresVectors,2)-1);
test.lbl = featuresVectors(ts,size(featuresVectors,2));

%Addestramento del classificatore sui soli dati di training
k = 1;
knn = fitcknn(training.features, training.lbl, 'Distance', 'euclidean', 'NumNeighbors', k, 'Standardize', 1);

%Test del classificatore
predicted_train = predict(knn,training.features);
predicted_test = predict(knn,test.features);

% Confusion matrix training
[cm_raw,order]=confusionmat(table2array(training.lbl(:,1)),predicted_train(:));
trainperf.cm_raw = array2table(cm_raw);
trainperf.cm_raw.Properties.VariableNames = order;
trainperf.cm = array2table(cm_raw ./ repmat(sum(cm_raw,2), 1, size(cm_raw,2)));
trainperf.cm.Properties.VariableNames = order;
trainperf.accuracy = sum(diag(cm_raw)) / numel(predicted_train);

% Confusion matrix test
[cm_raw,order]=confusionmat(table2array(test.lbl(:,1)),predicted_test(:));
perf.cm_raw = array2table(cm_raw);
perf.cm_raw.Properties.VariableNames = order;
perf.cm = array2table(cm_raw ./ repmat(sum(cm_raw,2), 1, size(cm_raw,2)));
perf.cm.Properties.VariableNames = order;
perf.accuracy = sum(diag(cm_raw)) / numel(predicted_test);