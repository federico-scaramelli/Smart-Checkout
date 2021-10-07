% createDataSet
% È richiesta da parte dell'utente la selezione della cartella contenente
% le immagini grezze da segmentare per il DataSet. La funzione segmenta 
% ogni immagine in singoli oggetti, e ne salva le parti tagliate e le mask
% all'interno  della seconda cartella selezionata dall'utente. Viene poi 
% richiesto, opzionalmente, di inserire i nomi di tutte le classi delle 
% quali fan parte gli oggetti, in modo tale che il software possa agevolare 
% il lavoro di suddivisione in classi, iniziando a creare le cartelle 
% in cui inserire le immagini di output per creare la groundtruth.

clear all
close all

%Selezione della cartella contenente le immagini da segmentare
rawDataFolder = uigetdir('./', 'Select raw data folder');
%Selezione della cartella che conterrà le immagini segmentate
outDataFolder = uigetdir('./', 'Select output segmented image folder');
if ~exist(outDataFolder, 'dir')
    mkdir outDataFolder;
end
outMaskFolder = [outDataFolder '\masks']; %Cartella per le maschere
if ~exist(outMaskFolder , 'dir')
    mkdir (outMaskFolder);
end
outImagesFolder = [outDataFolder '\images']; %Cartella per le immagini
if ~exist(outImagesFolder , 'dir')
    mkdir (outImagesFolder);
end

files = dir([rawDataFolder '\*.jpg']); %Elenco files immagine
nFiles = size(files, 1);
%Visito ogni file
for i=1:1:nFiles
    fileName = files(i).name;
    filePath = [rawDataFolder '\' fileName];
    if ~exist(filePath, 'file')
       continue; 
    end

    disp(['Starting segmentation of ' filePath]);

    im = imread(filePath);

    %E ne estraggo le imgs degli objs presenti e le loro maschere 
    [cropped, masks] = segmentImageEdge(im);

    [~,name,~] = fileparts(filePath);
    nImages = size(cropped, 1); %Numero di objs
    %Salvo ogni oggetto estratto come immagine .jpg e le maschere in .mat
    %Le maschere serviranno a calcolare le features di forma.
    for j=1:1:nImages
        imwrite(cropped{j}, [outImagesFolder '\' name '_' num2str(j) '.jpg']);
        mask = masks{j};
        save([outMaskFolder '\' name '_' num2str(j) '_mask'], 'mask');
    end

end

%Richiesta da parte dell'utente di inserire i nomi di tutte le classi
classes = input('Write all classes comma-separated: ', 's');
classes = strsplit(classes, ',');
nClass = size(classes, 2); 
for i=1:1:nClass
   mkdir([outDataFolder '\images\' classes{i}]);
end

%qui potrei mettere codice per far comparire ogni immagine e far
%scegliere la classe all utente tramite dialog box

disp('OK! Now subdivide your segmented dataset by grouping it into folders corresponding to classes.');