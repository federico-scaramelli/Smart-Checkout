% testClassifier
% Richiede all'utente un classificatore addestrato e la cartella di test.
% Il classificatore classifica ogni oggetto contenuto nelle immagini di
% test dopo averne effettuato la segmentazione.
% Successivamente viene generata la matrice di confusione confrontando le 
% predizioni con la groundtruth di ogni immagine.

clear all 
close all

perfTot = 0;

%Richiesta del classificatore all'utente
[classifierFile,path] = uigetfile('.\*.mat', 'Select a trained knn classifier');
classifier_knn = load(fullfile(path, classifierFile)); %Carico dalla memoria il classificatore
classifier_knn = classifier_knn.knn;
if ~isa(classifier_knn, 'ClassificationKNN')
    error('The selected classifier is not suitable.');
end

%Richiesta della cartella contenente le immagini di test
testFolder = uigetdir('./', 'Select test data folder');
testSet = imageDatastore([testFolder '\images'], 'IncludeSubfolders', true, ...
        'FileExtensions', '.jpg');
    
%Elenco delle immagini 
images = testSet.Files;
n = numel(images);

%Visito ogni immagine raw
for i = 1 : n
    path = char(images{i});
    im = imread(path);
    [~,name,~] = fileparts(path); %nome del file
    
    gt = load([testFolder '\gt\' name '_gt.mat']); %Carico la gt dell'img
    gt = gt.gt;
    
    [cropped, masks] = segmentImageEdge(im); %Segmento l'immagine
    
    nImages = size(cropped, 1);
    for j = 1:1:nImages
        % Creazione delle cartelle da lavoro
        if ~exist([testFolder '\processed\' name '_processed\images'],'dir')
            mkdir([testFolder '\processed\' name '_processed\images']);
        end
        if ~exist([testFolder '\processed\' name '_processed\masks'],'dir')
            mkdir([testFolder '\processed\' name '_processed\masks']);
        end
        imwrite(cropped{j}, [testFolder '\processed\' name '_processed\images\' num2str(j) '.jpg'], 'jpg');
        mask = masks{j};
        save([testFolder '\processed\' name '_processed\masks\' num2str(j) '_mask'], 'mask');
    end
    
    %Calcolo delle features sulle immagini segmentate
    featuresVectors = computeFeatures([testFolder '\processed\' name '_processed\']);
    
    %Predizione delle classi da parte del classificatore
    predicted = predict(classifier_knn, featuresVectors);
    
    %Stampa risultati
    figure;
    for j=1:1:nImages
        subplot(ceil(sqrt(nImages)),ceil(sqrt(nImages)),j);
        imshow(cropped{j});
        title(predicted{j});
    end
    
    % Confusion matrix
    [cm_raw,order]=confusionmat(gt(:),predicted(:));
    % Numero di predizione corrette ed errate
    perf{i}.cm_raw = array2table(cm_raw);
    perf{i}.cm_raw.Properties.VariableNames = order;
    % Probabilità predizioni corrette ed errate. repmat(..) ritorna una matrice
    % della stessa dimensione di cm_raw con, per ogni riga, il numero totale di
    % oggetti della classe corrispondente a tale riga.
    perf{i}.cm = array2table(cm_raw ./ repmat(sum(cm_raw,2), 1, size(cm_raw,2)));
    perf{i}.cm.Properties.VariableNames = order;
    % Accuratezza del classificatore data dal numero di predizione corrette
    % fratto il numero di elementi nella groundtruth (oggetti totali)
    perf{i}.accuracy = sum(diag(cm_raw)) / numel(gt);
    
    perfTot = perf{i}.accuracy + perfTot;
end

%Performance media
perf{n+1} = perfTot / n;