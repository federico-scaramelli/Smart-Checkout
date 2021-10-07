% Function computeFeatures() 
% INPUT:  dataFolder, cartella contenente le immagini segmentate sulle quali
%         calcolare i descrittori.
% OUTPUT: featuresVectors, tabella contenente nelle prime colonne i
%         valori delle features relative all'oggetto associato ad ogni 
%         riga; e nell'ultima colonna i nomi delle rispettive etichette, 
%         ovvero i nomi delle classi di cui fanno parte gli oggetti di ogni 
%         riga (la groundtruth). 
%         Nel caso in cui questa funzione dovesse essere chiamata da 
%         classifyObjsImg(), la 12esima colonna non rappresenterebbe la
%         groundtruth.

function featuresVectors = computeFeatures(dataFolder)
    if ~exist(dataFolder, 'dir')
        error('La cartella del dataSet non esiste.');
    end
    
    %Creazione del dataset
    dataSet = imageDatastore([dataFolder '\images'], 'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames', 'FileExtensions', '.jpg');
    %Stesso numero di osservazioni in ogni classe per il training
    nObjLbl = countEachLabel(dataSet);
    minSetCount = min(nObjLbl{:,2}); 
    dataSet = splitEachLabel(dataSet, minSetCount, 'randomize');

    %Elenco delle immagini 
    images = dataSet.Files;
    n = numel(images);
    labels = dataSet.Labels;

    meancol = []; 
    eccent = [];
    areaper = [];
    circ = [];
    majorAx = [];

    %Visito ogni immagine
    for j = 1 : n
      path = char(images{j});
      disp(['Compute features for image in ' path]);
      im = imread(path);
      [~,name,~] = fileparts(path);
      mask = [dataFolder '\masks\' name '_mask.mat'];
      mask = load(mask);
      mask = mask.mask;

      meancol = [meancol; compute_meancol(im)];
      eccent = [eccent; compute_eccent(mask)];
      areaper = [areaper; compute_areaper(mask)];
      majorAx = [majorAx; compute_majorAx(mask)];
      circ = [circ; compute_circ(mask)];
    end
    
    featuresVectors = num2cell([meancol, eccent, areaper, majorAx, circ]);
    featuresVectors = [featuresVectors, cellstr(labels)]; 
    featuresVectors = cell2table(featuresVectors);
 featuresVectors.Properties.VariableNames = {'MeanR' 'MeanG' 'MeanB'...
          'Eccentricity' 'AreaPer' 'MajorAxisLength' 'Circolarity' 'Label'};
end

function meancol = compute_meancol(im)
    im = im2double(im);
    r = im(:,:,1);
    g = im(:,:,2);
    b = im(:,:,3);
    n = sum(sum(and((r~=0), and((g~=0), (b~=0)))));
    meanr = sum(sum(r) / n);
    meang = sum(sum(g) / n);
    meanb = sum(sum(b) / n);
    meancol = [meanr meang meanb];
end

function eccent = compute_eccent(mask)
    measurements = regionprops(mask, 'Eccentricity');
    eccent = measurements.Eccentricity;
end

function circ = compute_circ(mask)
    measurements = regionprops(mask, 'Area', 'Perimeter');
    circ = (4 * pi * measurements.Area) ./ (measurements.Perimeter .^ 2);
end

function majorAx = compute_majorAx(mask)
    measurements = regionprops(mask, 'MajorAxisLength');
    majorAx = measurements.MajorAxisLength;
end

function areaper = compute_areaper(mask)
    measurements = regionprops(mask, 'Area', 'Perimeter');
    areaper = measurements.Area / measurements.Perimeter;
end