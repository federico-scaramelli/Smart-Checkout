% Addetsra e testa un classificatore in modalità cross-validation. 
% Usato per scoprire il modello finale di classificazione da utilizzare, 
% iterando su nuovi dati e parametri.

clear all 
close all

dataFolder = uigetdir('./', 'Select training data folder');

%Calcolo delle features per ogni immagine segmentata
featuresVectors = computeFeatures(dataFolder);

numObservation = size(featuresVectors,2);
features = featuresVectors(:,1:numObservation-1); %Descrittori di ogni oggetto (features)
lbl = featuresVectors(:,numObservation); %Groundtruth

disp('Creating classifier');

k = 1;
n = input('Choose number of folds n>1: ');
if (isempty(n))
   n = 10;
end
classifier_knn = fitcknn(features, lbl, 'Distance', 'euclidean', ...
    'NumNeighbors', k, 'KFold', n, 'Standardize', 1);

disp('Classifier created');


[predicted] = kfoldPredict(classifier_knn);

gt = table2cell(lbl);

% Confusion matrix
[cm_raw,order]=confusionmat(gt(:),predicted(:));
% Numero di predizione corrette ed errate
perf.cm_raw = array2table(cm_raw);
perf.cm_raw.Properties.VariableNames = order;
% Probabilità predizioni corrette ed errate. repmat(..) ritorna una matrice
% della stessa dimensione di cm_raw con, per ogni riga, il numero totale di
% oggetti della classe corrispondente a tale riga.
perf.cm = array2table(cm_raw ./ repmat(sum(cm_raw,2), 1, size(cm_raw,2)));
perf.cm.Properties.VariableNames = order;
% Accuratezza del classificatore data dal numero di predizione corrette
% fratto il numero di elementi nella groundtruth (oggetti totali)
perf.accuracy = sum(diag(cm_raw)) / numel(gt);