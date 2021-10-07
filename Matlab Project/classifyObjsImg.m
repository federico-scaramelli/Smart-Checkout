% classifyObjsImg
% Viene richiesta all'utente la selezione di un file immagine del quale
% effettuare la classificazione degli oggetti contenuti al suo interno. Il
% software segmenta l'immagine e la partiziona in sotto-immagini contenenti
% i singoli oggetti isolati durante il processo di segmentazione. Tali
% sotto-immagini vengono memorizzate in memoria e per ognuna vengono
% calcolate le features che la descrivono tramite computeFeatures(). Viene
% poi caricato dalla memoria un classificatore, sul quale viene chiamata la
% funzione predict(), che ritorna in output la predizione del classificatore
% riguardo l'immagine dell'oggetto che si sta tentando di classificare.

clear all 
close all

[file,path] = uigetfile('.\*.jpg', 'Select an image file to classify');
filePath = fullfile(path,file);
[~,fileName,~] = fileparts(file);

%Creazione delle cartelle di lavoro
if ~exist(['.\processed\' fileName '_processed'],'dir')
    mkdir(['.\processed\' fileName '_processed']);
end
if ~exist(['.\processed\' fileName '_processed\images'],'dir')
    mkdir(['.\processed\' fileName '_processed\images']);
end
if ~exist(['.\processed\' fileName '_processed\masks'],'dir')
    mkdir(['.\processed\' fileName '_processed\masks']);
end

im = imread(filePath);
[cropped, masks] = segmentImageEdge(im); %Segmento l'immagine

nImages = size(cropped, 1); 
for j=1:1:nImages
    imwrite(cropped{j}, ['.\processed\' fileName '_processed\images\' num2str(j) '.jpg'])
    mask = masks{j};
    save(['.\processed\' fileName '_processed\masks\' num2str(j) '_mask'], 'mask');
end


%Calcolo delle features per ogni immagine segmentata
featuresVectors = computeFeatures(['.\processed\' fileName '_processed']);

%Richiesta del classificatore all'utente
[classifierFile,path] = uigetfile('.\*.mat', 'Select a trained knn classifier');
classifier_knn = load(fullfile(path, classifierFile)); %Carico dalla memoria il classificatore
classifier_knn = classifier_knn.knn;

%Se il classificatore selezionato non è un classificatore knn non
%partizionato il programma ritorna errore.
if ~isa(classifier_knn, 'ClassificationKNN')
    error('The selected classifier is not a ClassificationKNN instance.');
end

%Predizione del classificatore
predicted = predict(classifier_knn, featuresVectors);

%Output
figure;
for i=1:1:nImages
    subplot(ceil(sqrt(nImages)),ceil(sqrt(nImages)),i);
    imshow(cropped{i});
    title(predicted{i});
end